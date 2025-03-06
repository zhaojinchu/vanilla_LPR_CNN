import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, annotation_file, target_size=(224, 224)):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_size = target_size
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.normpath(os.path.join(self.root_dir, row['image_path'])).replace("\\", "/")
        print(img_path)
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        # Normalize bounding box coordinates
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values
        
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        
        x_min_resized = int(x_min * scale_x)
        y_min_resized = int(y_min * scale_y)
        x_max_resized = int(x_max * scale_x)
        y_max_resized = int(y_max * scale_y)
        
        # Apply transformations
        resized_image = self.transform(image)
        original_image_tensor = transforms.ToTensor()(image)
        bbox_resized = torch.tensor([x_min_resized, y_min_resized, x_max_resized, y_max_resized], dtype=torch.float32)
        bbox_original = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        return original_image_tensor, resized_image, bbox_original, bbox_resized

# Define paths and target image size
data_dir = os.path.normpath("C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN")
annotations_file = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN/combined_dataset/annotations.csv"
target_size = (640, 480)  # Configurable image size

# Load dataset
dataset = LicensePlateDataset(data_dir, annotations_file, target_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Function to visualize original and resized images with bounding boxes
def visualize_samples(original_image_tensor, resized_image, bbox_original, bbox_resized):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert to PIL images
    original_image = transforms.ToPILImage()(original_image_tensor.squeeze(0))
    resized_image = transforms.ToPILImage()(resized_image.squeeze(0))
    
    # Draw bounding boxes
    draw_original = ImageDraw.Draw(original_image)
    draw_original.rectangle([bbox_original[0], bbox_original[1], bbox_original[2], bbox_original[3]], outline="red", width=3)
    
    draw_resized = ImageDraw.Draw(resized_image)
    draw_resized.rectangle([bbox_resized[0], bbox_resized[1], bbox_resized[2], bbox_resized[3]], outline="red", width=3)
    
    # Display images
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image with Bounding Box")
    axes[0].axis("off")
    
    axes[1].imshow(resized_image)
    axes[1].set_title("Resized Image with Bounding Box")
    axes[1].axis("off")
    
    plt.show()

# Fetch a sample and display it
original_image_tensor, resized_image, bbox_original, bbox_resized = next(iter(dataloader))
visualize_samples(original_image_tensor, resized_image, bbox_original.squeeze(0), bbox_resized.squeeze(0))
