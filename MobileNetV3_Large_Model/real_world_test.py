import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models


# Letterbox Function
def letterbox_image(image, target_size):
    """
    Resize an image while maintaining aspect ratio using letterboxing.
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_width = int(orig_w * scale)
    new_height = int(orig_h * scale)
    
    # Resize with bicubic
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Create black‐padded target image
    new_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_width) // 2
    paste_y = (target_h - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image


# Invert Letterbox Function
def invert_letterbox_bbox(
    pred_xmin, pred_ymin, pred_xmax, pred_ymax,
    orig_w, orig_h,
    target_w=640, target_h=480
):
    """
    Convert a bounding box from letterboxed coords (target_w x target_h)
    back to the original image coords (orig_w x orig_h).
    """
    scale = min(target_w / orig_w, target_h / orig_h)
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)

    pad_x = (target_w - scaled_w) // 2
    pad_y = (target_h - scaled_h) // 2

    # Remove the padding offset
    pred_xmin -= pad_x
    pred_xmax -= pad_x
    pred_ymin -= pad_y
    pred_ymax -= pad_y

    # Undo the scale
    pred_xmin = int(pred_xmin / scale)
    pred_ymin = int(pred_ymin / scale)
    pred_xmax = int(pred_xmax / scale)
    pred_ymax = int(pred_ymax / scale)

    return pred_xmin, pred_ymin, pred_xmax, pred_ymax


# No‐Annotation Dataset
class NoAnnotationDataset(Dataset):
    def __init__(self, image_dir, transform=None, target_size=(640, 480)):
        """
        image_dir: path to a directory of images
        transform: optional TorchVision transform(s) to apply AFTER letterboxing
        target_size: (width, height) for letterboxing
        """
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
               and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load original image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Letterbox to (640 x 480)
        letterboxed = letterbox_image(image, self.target_size)

        # Convert to tensor
        if self.transform:
            letterboxed_tensor = self.transform(letterboxed)
        else:
            letterboxed_tensor = transforms.ToTensor()(letterboxed)

        return letterboxed_tensor, (orig_w, orig_h), img_path


# Model Definition
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
            nn.Linear(128, 4)  # (x_min, y_min, x_max, y_max)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        bbox_preds = self.reg_head(x)
        return bbox_preds  # absolute coords in the 640x480 letterboxed space


# Inference
def main():
    # Config
    image_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN/real_life_data"
    model_path = "weights/best_model_epoch_8.pth" # CHANGE ME
    batch_size = 1
    num_images_to_test = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = NoAnnotationDataset(
        image_dir=image_dir,
        transform=None,         # e.g. transforms.Compose([...]) if needed
        target_size=(640, 480)  # match the model's input size
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model = LicensePlateDetector(pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return
    
    model.eval()

    # Inference Loop
    for i, (letterboxed_tensor, (orig_w, orig_h), img_path) in enumerate(data_loader):
        if i >= num_images_to_test:
            break
        
        letterboxed_tensor = letterboxed_tensor.to(device)

        with torch.no_grad():
            # shape: (1, 4)
            pred_bbox_letterboxed = model(letterboxed_tensor).cpu().squeeze(0)
        
        lxmin, lymin, lxmax, lymax = pred_bbox_letterboxed.tolist()

        # Invert the letterbox
        pxmin, pymin, pxmax, pymax = invert_letterbox_bbox(
            lxmin, lymin, lxmax, lymax,
            orig_w, orig_h,
            640, 480
        )

        # Draw on original
        original_img = Image.open(img_path[0]).convert("RGB")
        draw = ImageDraw.Draw(original_img)
        draw.rectangle([pxmin, pymin, pxmax, pymax], outline="red", width=3)

        # Display the result
        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f"Predicted Box on {os.path.basename(img_path[0])}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
