import clip
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP normalization
])

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load CLIP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def extract_features(data_loader):
    all_features = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        with torch.no_grad():
            features = clip_model.encode_image(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)

# Extract features from train and test datasets
test_features, test_labels = extract_features(test_loader)

# Load the saved model, must be paired to your CLIP model
loaded_classifier = joblib.load("./models/cifar100_clip_vit14_classifier.pkl")
# Predict and evaluate on the test set
test_preds = loaded_classifier.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_preds)

#Test accuracy with ViT-L/14: 0.8478
print(f'Test accuracy (loaded model): {test_accuracy}')
