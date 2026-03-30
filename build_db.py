import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from features import extract_lbp, extract_sfgd, extract_color_features
from models import SimpleNN, DeepNN, CNNFeatureExtractor

def get_image_numpy(dataset, index):
    img_tensor, label = dataset[index]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    if img_np.shape[2] == 1:
        # Grayscale image (MNIST)
        img_np_color = (img_np * 255).astype(np.uint8).squeeze(-1)
        img_np_gray = img_np_color
    else:
        # RGB image (CIFAR-10)
        img_np_color = (img_np * 255).astype(np.uint8)
        import cv2
        img_np_gray = cv2.cvtColor(img_np_color, cv2.COLOR_RGB2GRAY)
        
    # Return BOTH formats so we don't destroy color data
    return img_np_color, img_np_gray, label

def build_feature_database(dataset, dataset_name, num_samples=1000):
    print(f"--- Building Database for {dataset_name} ---")
    db = {'labels': [], 'lbp': [], 'nn': [], 'dnn': [], 'cnn': [], 'sfgd': [], 'color': []}
    
    sample_tensor, _ = dataset[0]
    flat_size = sample_tensor.numel() 
    
    nn_model = SimpleNN(input_dim=flat_size)
    dnn_model = DeepNN(input_dim=flat_size)
    cnn_model = CNNFeatureExtractor()
    
    # Optional but recommended for ResNet: ImageNet Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for i in tqdm(range(num_samples), desc="Extracting Features"):
        # Unpack BOTH color and grayscale versions
        img_np_color, img_np_gray, label = get_image_numpy(dataset, i)
        img_tensor = dataset[i][0].unsqueeze(0)
        
        if img_tensor.shape[1] == 1:
            img_tensor_rgb = img_tensor.repeat(1, 3, 1, 1)
        else:
            img_tensor_rgb = img_tensor
            
        # Apply normalization for the CNN
        img_tensor_cnn = normalize(img_tensor_rgb[0]).unsqueeze(0)
        
        db['labels'].append(label)
        # Pass gray to classical, color to color
        db['lbp'].append(extract_lbp(img_np_gray))
        db['sfgd'].append(extract_sfgd(img_np_gray))
        db['color'].append(extract_color_features(img_np_color))
        
        db['nn'].append(nn_model(img_tensor))
        db['dnn'].append(dnn_model(img_tensor))
        db['cnn'].append(cnn_model(img_tensor_cnn)) # Use normalized tensor
    
    with open(f"{dataset_name}_features.pkl", 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved {num_samples} images to {dataset_name}_features.pkl!\n")
   
if __name__ == "__main__":

    # 1. Define transforms for Grayscale and RGB datasets
    transform_gray = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    transform_rgb = transforms.Compose([transforms.ToTensor()])
    
    # 2. Download/Load all three datasets
    print("Loading datasets into memory...")
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    fashion_mnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb)
    
    # 3. Create a list of datasets to process
    datasets_to_process = [
        (mnist, "MNIST"),
        (fashion_mnist, "FashionMNIST"),
        (cifar10, "CIFAR10")
    ]
    
    # 4. Loop through each dataset and build its database!
    # (Leaving num_samples at 1000 for speed, but you can increase it later if you want)
    for dataset, name in datasets_to_process:
        build_feature_database(dataset, name, num_samples=1000)