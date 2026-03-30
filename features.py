import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct

# TASK 1: Classical Retrieval using LBP
def extract_lbp(image_gray, radius=2, n_points=16):
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# TASK 3: CNN Feature Extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.extractor(x)
        return features.view(features.size(0), -1).numpy().flatten()

# TASK 4: Novel Feature Representation (SFGD)
def extract_sfgd(image_gray, patch_size=7, keep_coeffs=4):
    h, w = image_gray.shape
    features = []
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = magnitude[i:i+patch_size, j:j+patch_size]
            if patch.shape != (patch_size, patch_size): continue
            dct_patch = dct(dct(patch.T, norm='ortho').T, norm='ortho')
            block_size = int(np.sqrt(keep_coeffs))
            low_freq = dct_patch[:block_size, :block_size].flatten()
            features.extend(low_freq)
            
    features = np.array(features)
    if np.linalg.norm(features) > 0:
        features = features / np.linalg.norm(features)
    return features

# TASK 5: Hybrid Retrieval Model
def fuse_features(f_cnn, f_proposed, method='concat', lam=0.5):
    """
    Fuses CNN features and Proposed features.
    method: 'concat' (Eq 5) or 'weighted' (Eq 6)
    lam: lambda weight for weighted addition
    """
    # Normalize features first to ensure equal weighting
    f_cnn = f_cnn / (np.linalg.norm(f_cnn) + 1e-8)
    f_proposed = f_proposed / (np.linalg.norm(f_proposed) + 1e-8)
    
    if method == 'concat':
        return np.concatenate((f_cnn, f_proposed))
    elif method == 'weighted':
        # Pad the smaller vector with zeros so they can be added mathematically
        max_len = max(len(f_cnn), len(f_proposed))
        f_cnn_pad = np.pad(f_cnn, (0, max_len - len(f_cnn)))
        f_prop_pad = np.pad(f_proposed, (0, max_len - len(f_proposed)))
        return (lam * f_cnn_pad) + ((1 - lam) * f_prop_pad)

# TASK 6: Inter- and Intra-Color Features
import numpy as np

def extract_color_features(img_np):
    """
    Extracts Intra-color and Inter-color features as strictly defined in Task 6.
    """
    # Safety check: If the image is grayscale (like MNIST), color features don't exist.
    # We return an array of 9 zeros so the math doesn't crash during fusion.
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return np.zeros(9)
    
    # Flatten the 2D color channels into 1D arrays for easy math
    R = img_np[:, :, 0].flatten()
    G = img_np[:, :, 1].flatten()
    B = img_np[:, :, 2].flatten()
    
    # --- 1. Calculate f_intra (Mean and Standard Deviation) ---
    mu_R, sigma_R = np.mean(R), np.std(R)
    mu_G, sigma_G = np.mean(G), np.std(G)
    mu_B, sigma_B = np.mean(B), np.std(B)
    
    f_intra = [mu_R, sigma_R, mu_G, sigma_G, mu_B, sigma_B]
    
    # --- 2. Calculate f_inter (Pearson Correlation Coefficient) ---
    # We use np.corrcoef. It returns a 2x2 matrix, we grab the off-diagonal [0, 1]
    # We also add a tiny safety check in case an image is perfectly blank (std = 0)
    def calc_corr(ch1, ch2):
        if np.std(ch1) == 0 or np.std(ch2) == 0:
            return 0.0
        return np.corrcoef(ch1, ch2)[0, 1]
        
    corr_RG = calc_corr(R, G)
    corr_GB = calc_corr(G, B)
    corr_RB = calc_corr(R, B)
    
    f_inter = [corr_RG, corr_GB, corr_RB]
    
    # --- 3. Combine into final feature vector ---
    # This creates a final list of exactly 9 numbers
    f_color = np.array(f_intra + f_inter)
    
    return f_color