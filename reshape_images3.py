import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def letterbox_image(image, target_size):
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
    def __init__(self, root_dir, annotation_file, target_size=(640, 480)):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.target_size = target_size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        # Force selection of the correct image from the annotation file
        img_path_fixed = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN/combined_dataset/images/0a0a00b2fbe89a47.jpg"
        
        # Select the row with this image
        # Normalize path before comparing
        normalized_path = os.path.normpath("combined_dataset/images/0a0a00b2fbe89a47.jpg").replace("\\", "/")
        row = self.annotations[self.annotations['image_path'].str.replace("\\", "/") == normalized_path]

        if row.empty:
            raise ValueError(f"No annotation found for {img_path_fixed}")
        
        row = row.iloc[0]  # Ensure we only take the first matching row
        image = Image.open(img_path_fixed).convert("RGB")
        original_width, original_height = image.size
        
        
        """
        row = self.annotations.iloc[idx]
        img_path = os.path.normpath(os.path.join(self.root_dir, row['image_path'])).replace("\\", "/")
        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size
        """
        
        # Bounding box coordinates
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values
        
        print(x_min, y_min, x_max, y_max)
        
        # Apply letterbox padding
        #resized_image, scale, pad_x, pad_y = letterbox_image(image, self.target_size)
        
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

        
        print(f"Original Size: {original_width}x{original_height}, Target Size: {self.target_size[0]}x{self.target_size[1]}")
        print(f"Scale: {scale}, Pad X: {pad_x}, Pad Y: {pad_y}")
        print(f"Original BBox: ({x_min}, {y_min}, {x_max}, {y_max})")

        
        # Convert to tensor
        resized_image_tensor = self.transform(resized_image)
        original_image_tensor = self.transform(image)
        bbox_resized = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        bbox_original = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        return original_image_tensor, resized_image_tensor, bbox_original, bbox_resized

# Define paths and target image size
data_dir = os.path.normpath("C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN")
annotations_file = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN/combined_dataset/annotations.csv"
target_size = (640, 480)  # Maintain aspect ratio with padding

# Load dataset
dataset = LicensePlateDataset(data_dir, annotations_file, target_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Function to visualize original and resized images with bounding boxes
def visualize_samples(original_image_tensor, resized_image_tensor, bbox_original, bbox_resized):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert to PIL images
    original_image = transforms.ToPILImage()(original_image_tensor.squeeze(0))
    resized_image = transforms.ToPILImage()(resized_image_tensor.squeeze(0))
    
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
    axes[1].set_title("Resized Image with Bounding Box (Letterboxed)")
    axes[1].axis("off")
    
    plt.show()

# Fetch a sample and display it
original_image_tensor, resized_image_tensor, bbox_original, bbox_resized = next(iter(dataloader))
visualize_samples(original_image_tensor, resized_image_tensor, bbox_original.squeeze(0), bbox_resized.squeeze(0))
