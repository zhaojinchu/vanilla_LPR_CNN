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
        
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        
        # Apply transformations
        image = self.transform(image)
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        return image, bbox

# Define paths and target image size
data_dir = os.path.normpath("C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN")
annotations_file = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN/combined_dataset/annotations.csv"
target_size = (640, 480)  # Configurable image size

# Load dataset
dataset = LicensePlateDataset(data_dir, annotations_file, target_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Function to visualize an image with bounding box
def visualize_sample(image, bbox):
    image = transforms.ToPILImage()(image.squeeze(0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Fetch a sample and display it
image, bbox = next(iter(dataloader))
visualize_sample(image, bbox.squeeze(0))
