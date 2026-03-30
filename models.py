import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# --- Task 2: Standard NN and Deep NN ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim=28*28):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flattens the image
        return self.fc(x).detach().numpy().flatten()

class DeepNN(nn.Module):
    def __init__(self, input_dim=28*28):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x).detach().numpy().flatten()

# --- Task 3: CNN ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # For grayscale (1 channel), ResNet expects 3. We handle this in build_db.py
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.extractor(x)
        return features.view(features.size(0), -1).numpy().flatten()