import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def letterbox_image(image, target_size):
    """
    Resize an image while maintaining aspect ratio using letterboxing.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # Create a new black image with the target size
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image, scale, paste_x, paste_y, new_width, new_height

class LicensePlateDataset(Dataset):
    """
    Custom PyTorch Dataset for License Plate Detection.
    Loads images, applies letterbox resizing, and processes bounding boxes.
    """
    def __init__(self, root_dir, annotation_file, target_size=(640, 480), transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_size = target_size
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        img_path = os.path.normpath(img_path).replace("\\", "/")
        
        print(f"Loading image: {img_path}")  # Debugging
        
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size
        
        # Extract bounding box
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)
        
        # Apply letterbox padding
        resized_image, scale, pad_x, pad_y, new_width, new_height = letterbox_image(image, self.target_size)
        
        # First, scale the bounding box coordinates
        x_min = int(x_min * scale)
        y_min = int(y_min * scale)
        x_max = int(x_max * scale)
        y_max = int(y_max * scale)

        # Then, apply padding based on the letterboxing direction
        if new_width < self.target_size[0]:  # Letterboxing on left/right (height was scaled)
            x_min += pad_x
            x_max += pad_x
        elif new_height < self.target_size[1]:  # Letterboxing on top/bottom (width was scaled)
            y_min += pad_y
            y_max += pad_y

        # Convert to tensors
        resized_image_tensor = self.transform(resized_image)
        bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return resized_image_tensor, bbox_tensor

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

        # Modify first layer for input size (640x480)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove classification head
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  

        # Custom regression head for bounding box detection
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  # Output: (x_min, y_min, x_max, y_max)
            nn.Sigmoid()  # Normalize to [0,1]
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x  # Bounding box coordinates in normalized format
    
# Configuration
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
annotations_file = os.path.join(data_dir, "combined_dataset/annotations.csv")
target_size = (640, 480)

# Create dataset and dataloader
dataset = LicensePlateDataset(data_dir, annotations_file, target_size)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, persistent_workers = True)

# Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.synchronize()

model = LicensePlateDetector(backbone="resnet18", pretrained=True).to(device)


# Loss Function and Optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training Configuration
num_epochs = 100
best_val_loss = float("inf")
save_dir = "weights"

# Start Training Loop
if __name__ == "__main__":
    print(f"Using device: {device}", flush=True)
    
    for epoch in range(num_epochs):
        print("Another Epoch...")
        model.train()
        running_loss = 0.0

        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            # Normalize targets (since model outputs 0-1 range)
            targets = targets / torch.tensor([640, 480, 640, 480], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}", flush=True)

        # Save model if validation loss improves
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved: {model_path} (Loss: {epoch_loss:.4f})", flush=True)

        # Debugging: Print Sample Predictions Every 5 Epochs
        if (epoch + 1) % 5 == 0:
            sample_images, sample_targets = next(iter(dataloader))
            sample_images = sample_images.to(device)
            sample_preds = model(sample_images)

            # Convert predictions back to absolute coordinates
            sample_preds = sample_preds * torch.tensor([640, 480, 640, 480], dtype=torch.float32).to(device)
            sample_targets = sample_targets.to(device)

            print(f"🔍 Sample Predictions (Epoch {epoch+1}):\n"
                  f"📌 Pred: {sample_preds[0].cpu().detach().numpy()}\n"
                  f"🎯 Target: {sample_targets[0].cpu().numpy()}\n", flush=True)
            
    print("Training Complete!")
