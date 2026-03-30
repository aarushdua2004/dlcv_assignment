import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from features import fuse_features

def calculate_metrics(retrieved_labels, query_label, total_relevant):
    """Calculates Precision@K, Recall@K, and Average Precision."""
    retrieved_labels = np.array(retrieved_labels)
    relevant_mask = (retrieved_labels == query_label)
    
    # Precision = Relevant Retrieved / Total Retrieved
    precision = np.sum(relevant_mask) / len(retrieved_labels)
    
    # Recall = Relevant Retrieved / Total Relevant in Database
    # (If total_relevant is 0, recall is 0 to avoid division by zero)
    recall = np.sum(relevant_mask) / total_relevant if total_relevant > 0 else 0
    
    # Average Precision (AP)
    ap = 0
    relevant_count = 0
    for i, is_relevant in enumerate(relevant_mask):
        if is_relevant:
            relevant_count += 1
            ap += relevant_count / (i + 1)
    ap = ap / total_relevant if total_relevant > 0 else 0
    
    return precision, recall, ap

def evaluate_system(db_path, k=10, num_queries=50):
    print(f"--- Evaluating Database: {db_path} ---")
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
        
    labels = np.array(db['labels'])
    
    # ADDED Task 6 Methods to the evaluation list
    methods = ['lbp', 'nn', 'dnn', 'cnn', 'sfgd', 'hybrid', 'color', 'color+lbp', 'color+cnn']
    
    # To store mean metrics
    results = {m: {'precision': [], 'recall': [], 'ap': []} for m in methods}
    
    # Pick random indices to act as our "Query Images"
    np.random.seed(42)
    query_indices = np.random.choice(len(labels), num_queries, replace=False)
    
    for q_idx in query_indices:
        q_label = labels[q_idx]
        total_relevant = np.sum(labels == q_label)
        
        for method in methods:
            if method == 'hybrid':
                # Create hybrid on the fly for query and database
                q_feat = fuse_features(db['cnn'][q_idx], db['sfgd'][q_idx])
                db_feats = [fuse_features(c, s) for c, s in zip(db['cnn'], db['sfgd'])]
                
            elif method == 'color+lbp':
                # Task 6: Fuse Color and LBP
                q_feat = fuse_features(db['color'][q_idx], db['lbp'][q_idx])
                db_feats = [fuse_features(c, l) for c, l in zip(db['color'], db['lbp'])]
                
            elif method == 'color+cnn':
                # Task 6: Fuse Color and CNN
                q_feat = fuse_features(db['color'][q_idx], db['cnn'][q_idx])
                db_feats = [fuse_features(c, cn) for c, cn in zip(db['color'], db['cnn'])]
                
            else:
                q_feat = db[method][q_idx]
                db_feats = db[method]
                
            # Reshape for sklearn
            q_feat = np.array(q_feat).reshape(1, -1)
            db_feats = np.array(db_feats)
            
            # Compute similarities
            similarities = cosine_similarity(q_feat, db_feats)[0]
            
            # Get Top-K indices (excluding the query image itself)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = [idx for idx in top_indices if idx != q_idx][:k]
            
            # Calculate Metrics
            retrieved_labels = labels[top_indices]
            p, r, ap = calculate_metrics(retrieved_labels, q_label, total_relevant)
            
            results[method]['precision'].append(p)
            results[method]['recall'].append(r)
            results[method]['ap'].append(ap)
            
    # Print the Comparative Table
    print(f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP':<10}")
    print("-" * 55)
    for method in methods:
        m_p = np.mean(results[method]['precision'])
        m_r = np.mean(results[method]['recall'])
        m_ap = np.mean(results[method]['ap'])
        print(f"{method.upper():<15} | {m_p:.4f}      | {m_r:.4f}    | {m_ap:.4f}")

if __name__ == "__main__":
    # List of all the databases your build_db.py creates
    databases = [
        "MNIST_features.pkl", 
        "FashionMNIST_features.pkl", 
        "CIFAR10_features.pkl"
    ]
    
    for db_file in databases:
        if os.path.exists(db_file):
            evaluate_system(db_file, k=10, num_queries=50)
            print("\n" + "="*60 + "\n") # Prints a visual separator between tables
        else:
            print(f"Could not find {db_file}. Please make sure it was built.")