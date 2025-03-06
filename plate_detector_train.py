import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

"""
Loading and Preprocessing Dataset
"""
# Dataset Class
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        height, width, _ = image.shape

        # Read bounding box
        x_min, y_min, x_max, y_max = row[1:].astype(float)

        # Scale image to 640x480 (keeping aspect ratio)
        new_width, new_height = 640, 480
        image = cv2.resize(image, (new_width, new_height))

        # Scale bounding box coordinates
        x_min = (x_min / width) * new_width
        y_min = (y_min / height) * new_height
        x_max = (x_max / width) * new_width
        y_max = (y_max / height) * new_height

        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        # Convert image to tensor manually (no Albumentations)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same as pretrained models
        ])

        image = transform(image)

        return image, bbox


# Define Data Augmentations
transform = A.Compose([
    A.Resize(640, 480),  # Resize to model input size
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Imagenet normalization
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Load Dataset
csv_file = "./processed_dataset/train_dataset.csv"
dataset = LicensePlateDataset(csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

"""
Building CNN Model
"""
class LicensePlateDetector(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super(LicensePlateDetector, self).__init__()

        # Choose backbone
        if backbone == "resnet18":
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.base_model.fc.in_features  
        elif backbone == "mobilenet":
            self.base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            in_features = self.base_model.classifier[-1].in_features
        else:
            raise ValueError("Invalid backbone! Choose 'resnet18' or 'mobilenet'.")

        # Modify first layer to handle 640x480
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove classification head
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  

        # Custom detection head
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x  # Bounding box coordinates


# Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LicensePlateDetector(backbone="resnet18", pretrained=True).to(device)

"""
Loss and Optimizers
"""
# Define Loss Function and Optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

"""
Training Loop
"""

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training Complete!")