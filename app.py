import streamlit as st
import pickle
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from collections import Counter

# Import your custom tools
from features import extract_lbp, extract_sfgd, extract_color_features, fuse_features
from models import CNNFeatureExtractor, SimpleNN, DeepNN

st.set_page_config(page_title="Next-Gen Image Retrieval", layout="wide")
st.title("Next-Generation Image Retrieval System")

# --- Class Name Mappings ---
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "FashionMNIST": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "CIFAR10": ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
}

# --- Load Datasets for Display ---
@st.cache_resource
def load_raw_datasets():
    fashion = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    return {"FashionMNIST": fashion, "MNIST": mnist, "CIFAR10": cifar}

raw_datasets = load_raw_datasets()

# --- Sidebar Controls ---
st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox("Select Database", ["FashionMNIST", "MNIST", "CIFAR10"])

@st.cache_data
def load_db(dataset_name):
    file_name = f"{dataset_name}_features.pkl"
    with open(file_name, 'rb') as f:
        return pickle.load(f)

try:
    db = load_db(dataset_choice)
    current_dataset = raw_datasets[dataset_choice]
except FileNotFoundError:
    st.error(f"Database file '{dataset_choice}_features.pkl' not found. Please run build_db.py first.")
    st.stop()

k_val = st.sidebar.number_input("Select Top-K (Max 20)", min_value=1, max_value=20, value=10)
method_choice = st.sidebar.selectbox("Feature Method", 
                                     ['LBP', 'NN', 'DNN', 'CNN', 'SFGD', 'Hybrid', 'Color', 'Color + LBP', 'Color + CNN'])

uploaded_file = st.sidebar.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"])

st.sidebar.info("System uses **Majority Voting** to identify the most likely class cluster.")

if uploaded_file is not None:
    # 1. Open and Resize
    raw_img = Image.open(uploaded_file)
    
    if dataset_choice == "CIFAR10":
        target_size = (32, 32)
        q_img_processed = raw_img.convert('RGB').resize(target_size)
    else:
        target_size = (28, 28)
        q_img_processed = raw_img.convert('L').resize(target_size)
    
    st.sidebar.image(q_img_processed, caption="Query Image (Resized)", width=150)
    
    # 2. Prepare Formats
    img_np = np.array(q_img_processed)
    # Tensor converts to [0, 1] scale automatically
    img_tensor = transforms.ToTensor()(q_img_processed).unsqueeze(0) 
    
    if dataset_choice == "CIFAR10":
        img_tensor_rgb = img_tensor
        classical_img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_tensor_rgb = img_tensor.repeat(1, 3, 1, 1)
        classical_img_np = img_np 

    # --- CRITICAL FIX: ImageNet Normalization for CNN ---
    # ResNet needs this specific normalization to "see" properly
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cnn_input = normalize(img_tensor_rgb[0]).unsqueeze(0)

    # 3. Extract Features
    try:
        with st.spinner(f"Searching {dataset_choice} using {method_choice}..."):
            if method_choice == 'LBP':
                q_feat = extract_lbp(classical_img_np)
                db_feats = db['lbp']
            elif method_choice == 'NN':
                q_feat = SimpleNN(input_dim=img_tensor.numel())(img_tensor)
                db_feats = db['nn']
            elif method_choice == 'DNN':
                q_feat = DeepNN(input_dim=img_tensor.numel())(img_tensor)
                db_feats = db['dnn']
            elif method_choice == 'CNN':
                # Replaced img_tensor_rgb with cnn_input
                q_feat = CNNFeatureExtractor()(cnn_input)
                db_feats = db['cnn']
            elif method_choice == 'SFGD':
                q_feat = extract_sfgd(classical_img_np)
                db_feats = db['sfgd']
            elif method_choice == 'Color':
                color_input = np.array(raw_img.convert('RGB').resize(target_size))
                q_feat = extract_color_features(color_input)
                db_feats = db['color']
            elif method_choice == 'Hybrid':
                # Replaced img_tensor_rgb with cnn_input
                f_cnn = CNNFeatureExtractor()(cnn_input)
                f_sfgd = extract_sfgd(classical_img_np)
                q_feat = fuse_features(f_cnn, f_sfgd)
                db_feats = [fuse_features(c, s) for c, s in zip(db['cnn'], db['sfgd'])]
            elif method_choice == 'Color + LBP':
                color_input = np.array(raw_img.convert('RGB').resize(target_size))
                f_color = extract_color_features(color_input)
                f_lbp = extract_lbp(classical_img_np)
                q_feat = fuse_features(f_color, f_lbp)
                db_feats = [fuse_features(c, l) for c, l in zip(db['color'], db['lbp'])]
            elif method_choice == 'Color + CNN':
                color_input = np.array(raw_img.convert('RGB').resize(target_size))
                f_color = extract_color_features(color_input)
                # Replaced img_tensor_rgb with cnn_input
                f_cnn = CNNFeatureExtractor()(cnn_input)
                q_feat = fuse_features(f_color, f_cnn)
                db_feats = [fuse_features(c, cn) for c, cn in zip(db['color'], db['cnn'])]

            # 4. Similarity and Majority Vote
            q_feat = np.array(q_feat).reshape(1, -1)
            db_feats = np.array(db_feats)
            similarities = cosine_similarity(q_feat, db_feats)[0]
            top_indices = np.argsort(similarities)[::-1][:k_val]

            # --- MAJORITY VOTING LOGIC ---
            retrieved_labels = [int(current_dataset[idx][1]) for idx in top_indices]
            vote_counts = Counter(retrieved_labels)
            majority_class, match_count = vote_counts.most_common(1)[0]
            
            class_name = CLASS_NAMES[dataset_choice][majority_class]
            precision_at_k = (match_count / k_val) * 100

            # 5. Display Metrics
            st.subheader(f"Results for Detected Cluster: {class_name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Class", class_name)
            m2.metric("Majority Count", f"{match_count}/{k_val}")
            m3.metric("Cluster Precision", f"{precision_at_k:.1f}%")

            st.markdown("---")
            
            # Grid Display
            cols_per_row = 5
            for i in range(0, k_val, cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx_in_top = i + j
                    if idx_in_top < k_val:
                        db_idx = top_indices[idx_in_top]
                        res_img, res_label = current_dataset[db_idx]
                        res_name = CLASS_NAMES[dataset_choice][res_label]
                        sim_score = similarities[db_idx]
                        with cols[j]:
                            st.image(res_img, use_container_width=True)
                            st.caption(f"**{res_name}**\nSim: {sim_score:.3f}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload an image to start.")