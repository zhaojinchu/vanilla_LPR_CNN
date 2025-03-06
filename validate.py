import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
from train import LicensePlateDetector  # Ensure this is correctly imported
from train import LicensePlateDataset  # Ensure this is correctly imported

# Configuration
model_path = "weights/best_model_epoch_38.pth"  # Replace with actual best model path
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
annotations_file = os.path.normpath(os.path.join(data_dir, "combined_dataset/annotations.csv")).replace("\\", "/")
target_size = (640, 480)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
val_dataset = LicensePlateDataset(data_dir, annotations_file, target_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = LicensePlateDetector(backbone="resnet18", pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Function to draw bounding boxes
def draw_bounding_box(image, bbox, color=(0, 255, 0), label="Pred"):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Run validation
for i, (image_tensor, target_bbox) in enumerate(val_loader):
    image_tensor = image_tensor.to(device)
    target_bbox = target_bbox.squeeze(0).cpu().numpy()

    # Get model prediction
    with torch.no_grad():
        pred_bbox = model(image_tensor).squeeze(0).cpu().numpy()
        pred_bbox = pred_bbox * np.array([640, 480, 640, 480])  # Convert to absolute coordinates

    # Convert tensor to image
    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw ground truth and predicted bounding boxes
    image_np = draw_bounding_box(image_np, target_bbox, color=(0, 0, 255), label="GT")
    image_np = draw_bounding_box(image_np, pred_bbox, color=(0, 255, 0), label="Pred")

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {i+1}: Red = Ground Truth, Green = Prediction")
    plt.axis("off")
    plt.show()
    
    if i == 9:  # Show only first 10 images for validation
        break
