import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis, parameter_count_table
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
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

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

# Validation metrics
correct_predictions = 0
total_samples = 0
threshold = 20  # Allowable pixel error threshold
inference_times = []

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Run validation
sampled_indices = random.sample(range(len(val_dataset)), min(50, len(val_dataset)))
for i in sampled_indices:
    image_tensor, target_bbox = val_dataset[i]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    target_bbox = target_bbox.cpu().numpy()

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        pred_bbox = model(image_tensor).squeeze(0).cpu().numpy()
    end_time = time.time()
    inference_times.append(end_time - start_time)
    
    pred_bbox = pred_bbox * np.array([640, 480, 640, 480])  # Convert to absolute coordinates

    iou_score = iou(pred_bbox, target_bbox)
    if iou_score > 0.5:  # IOU Threshold for correct prediction
        correct_predictions += 1
    total_samples += 1

    # Convert tensor to image
    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw ground truth and predicted bounding boxes
    image_np = draw_bounding_box(image_np, target_bbox, color=(0, 0, 255), label="GT")
    image_np = draw_bounding_box(image_np, pred_bbox, color=(0, 255, 0), label="Pred")

    # Display selected images
    if i % 5 == 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {i}: Red = Ground Truth, Green = Prediction")
        plt.axis("off")
        plt.show()

# Calculate and print accuracy
accuracy = (correct_predictions / total_samples) * 100
avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
throughput = 1.0 / np.mean(inference_times)  # Images per second

print(f"Model Accuracy: {accuracy:.2f}% over {total_samples} samples")
print(f"Average Inference Time: {avg_inference_time:.2f} ms per image")
print(f"Throughput: {throughput:.2f} images per second")

# Compute FLOPs and Parameters
sample_input = torch.randn(1, 3, 640, 480).to(device)
flops = FlopCountAnalysis(model, sample_input)
params = parameter_count_table(model)

print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print("Model Parameters:")
print(params)
