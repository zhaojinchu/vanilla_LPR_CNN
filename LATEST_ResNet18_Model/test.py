import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn

# Custom Dataset (Now Non-Iterable for Debugging)
class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = os.path.join(root_dir, "combined_dataset/preletterboxed")
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, os.path.basename(row['image_path']))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping corrupted image: {img_path} ({e})")
            return None  # Skip corrupted images

        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)
        image_tensor = self.transform(image)
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return image_tensor, bbox, img_path  # Return image path for visualization

# Load Model
class LicensePlateDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(LicensePlateDetector, self).__init__()
        
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  
        in_channels = 512

        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4)  # Predicting (x_min, y_min, x_max, y_max)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        bbox_preds = self.reg_head(x)
        return bbox_preds  # Returns bounding box predictions in absolute coordinates

# Configuration
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
val_dataset = LicensePlateDataset(data_dir, os.path.join(data_dir, "combined_dataset/val_annotations.csv"))

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)  # Debugging: One image at a time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LicensePlateDetector(pretrained=False).to(device)

# Load Best Model
model_path = "weights/best_model_epoch_8.pth"  # Change to the correct best model file
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")
else:
    print(f"❌ Model file not found: {model_path}")
    exit()

model.eval()

# Function to Draw Bounding Boxes
def draw_bounding_boxes(image_path, predicted_bbox, ground_truth_bbox):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw Ground Truth Box (Blue)
    draw.rectangle(ground_truth_bbox.tolist(), outline="blue", width=3)
    
    # Draw Predicted Box (Red)
    draw.rectangle(predicted_bbox.tolist(), outline="red", width=3)

    return image

# Inference and Visualization
num_test_images = 10  # Adjust as needed
for batch_idx, (image_tensor, target_bbox, img_path) in enumerate(val_loader):
    if batch_idx >= num_test_images:
        break
    
    print(img_path)
    print(target_bbox)
    
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output_bbox = model(image_tensor).cpu().squeeze(0)  # Get prediction
    print(output_bbox)
    # Extract Bounding Boxes
    ground_truth_bbox = target_bbox.squeeze(0)
    predicted_bbox = output_bbox

    # Draw Bounding Boxes on Image
    result_image = draw_bounding_boxes(img_path[0], predicted_bbox, ground_truth_bbox)

    # Show Image
    plt.figure(figsize=(6, 6))
    plt.imshow(result_image)
    plt.title(f"Prediction (Red) vs Ground Truth (Blue)")
    plt.axis("off")
    plt.show()
