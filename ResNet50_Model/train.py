import os
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import csv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes"""
    # Convert to float (ensure no overflows)
    boxA = np.array(boxA, dtype=np.float32)
    boxB = np.array(boxB, dtype=np.float32)

    # Ensure coordinates are within valid image range
    boxA = np.clip(boxA, 0, 640)  # Assuming image size is 640x480
    boxB = np.clip(boxB, 0, 640)

    # Calculate intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))  # Prevent zero area
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0  # Prevent division by zero

def calculate_metrics(iou_scores, iou_threshold=0.5):
    """Computes mAP, Precision, Recall, and F1-score based on IoU thresholds"""
    tp = sum(iou >= iou_threshold for iou in iou_scores)  # True Positives
    fp = sum(iou < iou_threshold for iou in iou_scores)   # False Positives
    fn = 0  # No False Negatives since we predict exactly 1 box per image
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_ap = np.mean(iou_scores)

    return mean_ap, precision, recall, f1_score

class LicensePlateIterableDataset(IterableDataset):
    def __init__(self, root_dir, annotation_file, transform=None, chunk_size=1024):
        self.root_dir = os.path.join(root_dir, "combined_dataset/preletterboxed")
        self.annotation_file = annotation_file
        self.transform = transform if transform else transforms.ToTensor()
        self.chunk_size = chunk_size

    def __iter__(self):
        annotations = pd.read_csv(self.annotation_file, chunksize=self.chunk_size)

        for chunk in annotations:
            for _, row in chunk.iterrows():
                img_path = os.path.join(self.root_dir, os.path.basename(row['image_path']))

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"⚠️ Skipping corrupted image: {img_path} ({e})")
                    continue

                x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)

                image_tensor = self.transform(image)
                bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

                yield image_tensor, bbox_tensor

class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(LicensePlateDetector, self).__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  
        in_channels = 2048  # ResNet-50 has 2048 channels in the final conv layer

        # Adjust first convolution layer to better handle 640x480 images
        self.feature_extractor[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Adjust stride lower to extract smaller details

        # Bounding box regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4)
        )
        
        """
        # OTHER MODEL TO TRY WITHOUT LINEAR - FULLY CONVOLUTIONAL
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Reduce to (batch, 128, 1, 1)
            nn.Conv2d(128, 4, kernel_size=1)  # Instead of Linear(128, 4)
        )
        """

    def forward(self, x):
        x = self.feature_extractor(x)
        bbox_preds = self.reg_head(x)   
        
        #return bbox_preds.view(x.shape[0], 4)
        return bbox_preds

# Training Config
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"

train_dataset = LicensePlateIterableDataset(data_dir, os.path.join(data_dir, "combined_dataset/train_annotations.csv"))
val_dataset = LicensePlateIterableDataset(data_dir, os.path.join(data_dir, "combined_dataset/val_annotations.csv"))

train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LicensePlateDetector(pretrained=True).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

num_epochs = 10
best_val_loss = float("inf")
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)

# CSV Logging
log_file = os.path.join(save_dir, "metrics.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "mAP", "precision", "recall", "F1_score", "FPS"])

# Training Loop
debug = False  # Set to False for full training
if __name__ == "__main__":
    print(f"Using device: {device}", flush=True)
    
    scaler = torch.amp.GradScaler()

    for epoch in range(1 if debug else num_epochs):  # Only 1 epoch for quick testing
        print(f"Epoch {epoch+1}/{num_epochs}...", flush=True)

        # Training Phase
        model.train()
        running_loss = 0.0
        iou_scores = []
        start_time = time.time()
        image_counter = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"🟢 Training Epoch {epoch+1}", unit="batch")):
            if debug and batch_idx >= 2:  # Stop early for quick testing
                break

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):  
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            image_counter += len(images)

            for i in range(len(outputs)):
                iou_scores.append(compute_iou(outputs[i].detach().cpu().numpy(), targets[i].cpu().numpy()))

        scheduler.step()
        train_loss = running_loss / (batch_idx + 1)  # Avoid division error
        fps = image_counter / (time.time() - start_time)
        mean_iou = np.mean(iou_scores)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou_scores = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"🔵 Validating Epoch {epoch+1}", unit="batch")):
                if debug and batch_idx >= 2:  # Stop early
                    break

                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                for i in range(len(outputs)):
                    val_iou_scores.append(compute_iou(outputs[i].cpu().numpy(), targets[i].cpu().numpy()))

        val_loss /= (batch_idx + 1)
        mean_val_iou = np.mean(val_iou_scores)
        mAP, precision, recall, f1_score = calculate_metrics(val_iou_scores)

        print(f"📊 Validation Loss: {val_loss:.4f}, IoU: {mean_val_iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        # Save metrics to CSV
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, mean_val_iou, mAP, precision, recall, f1_score, fps])

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved: {model_path} (Loss: {val_loss:.4f})")

    print("🎉 Debugging Complete! Metrics should be logged in metrics.csv")

