import os
import re
import csv
import glob
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def compute_iou(boxA, boxB):
    boxA = np.array(boxA, dtype=np.float32)
    boxB = np.array(boxB, dtype=np.float32)
    boxA = np.clip(boxA, 0, 640)
    boxB = np.clip(boxB, 0, 640)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0

def calculate_metrics(iou_scores, iou_threshold=0.5):
    tp = sum(iou >= iou_threshold for iou in iou_scores)
    fp = sum(iou < iou_threshold for iou in iou_scores)
    fn = 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_ap = np.mean(iou_scores) if len(iou_scores) > 0 else 0
    return mean_ap, precision, recall, f1_score

class LicensePlateIterableDataset(torch.utils.data.IterableDataset):
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
                    print(f"Skipping corrupted image: {img_path} ({e})")
                    continue
                x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)
                image_tensor = self.transform(image)
                bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
                yield image_tensor, bbox_tensor

class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(LicensePlateDetector, self).__init__()

        base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        base_model.classifier = nn.Identity()
        
        # Adjust first convolution layer to better handle 640x480 images
        first_block = base_model.features[0]   
        conv_layer = first_block[0]            
        conv_layer.stride = (1, 1)
        conv_layer.kernel_size = (7, 7)
        conv_layer.padding = (3, 3)

        self.feature_extractor = base_model.features 
        in_channels = 960
        
        # Bounding box regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4)
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

# Training Loop
if __name__ == "__main__":
    data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
    train_dataset = LicensePlateIterableDataset(
        data_dir, 
        os.path.join(data_dir, "combined_dataset/train_annotations.csv")
    )
    val_dataset = LicensePlateIterableDataset(
        data_dir, 
        os.path.join(data_dir, "combined_dataset/val_annotations.csv")
    )

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LicensePlateDetector(pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    num_epochs = 10
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    model_files = glob.glob(os.path.join(save_dir, "best_model_epoch_*.pth"))
    start_epoch = 0
    best_val_loss = float("inf")

    if model_files:
        epochs_found = []
        for mf in model_files:
            match = re.search(r"best_model_epoch_(\d+)\.pth", os.path.basename(mf))
            if match:
                epochs_found.append(int(match.group(1)))

        if epochs_found:
            last_epoch_found = max(epochs_found)
            best_model_file = os.path.join(save_dir, f"best_model_epoch_{last_epoch_found}.pth")
            print(f"Loading best model weights from {best_model_file}")
            model.load_state_dict(torch.load(best_model_file))
            start_epoch = last_epoch_found
        else:
            print("No valid best_model_epoch_*.pth files found. Starting from scratch.")
    else:
        print("No existing best model file found. Starting from scratch.")

    log_file = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "mAP", "precision", "recall", "F1_score", "FPS"])
    else:
        df = pd.read_csv(log_file)
        if len(df) > 0:
            last_logged_epoch = df["epoch"].iloc[-1]
            start_epoch = max(start_epoch, last_logged_epoch)
        print(f"Resuming from epoch {start_epoch + 1}")


    # Training Phase
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        iou_scores = []
        start_time = time.time()
        image_counter = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")):
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
                iou = compute_iou(outputs[i].detach().cpu().numpy(), targets[i].cpu().numpy())
                iou_scores.append(iou)

        scheduler.step()
        train_loss = running_loss / (batch_idx + 1)
        fps = image_counter / (time.time() - start_time)
        mean_iou = np.mean(iou_scores) if iou_scores else 0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou_scores = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", unit="batch")):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                for i in range(len(outputs)):
                    iou_val = compute_iou(outputs[i].cpu().numpy(), targets[i].cpu().numpy())
                    val_iou_scores.append(iou_val)

        val_loss /= (batch_idx + 1)
        mean_val_iou = np.mean(val_iou_scores) if val_iou_scores else 0
        mAP, precision, recall, f1_score = calculate_metrics(val_iou_scores)

        print(f"[Epoch {epoch+1}] val_loss: {val_loss:.4f}, IoU: {mean_val_iou:.4f}, "
              f"Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        # Logging Metrics
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                train_loss,
                val_loss,
                mean_val_iou,
                mAP,
                precision,
                recall,
                f1_score,
                fps
            ])

        # Saves best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Saved new best model: {best_model_path}")

    print("Training complete")
