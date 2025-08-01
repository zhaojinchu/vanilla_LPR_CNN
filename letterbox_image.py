import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

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
    
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image

# ✅ Paths
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
annotations_file = os.path.join(data_dir, "combined_dataset/annotations.csv")
output_dir = os.path.join(data_dir, "combined_dataset/preletterboxed")  # Directory for resized images

os.makedirs(output_dir, exist_ok=True)

# ✅ Load dataset annotations
annotations = pd.read_csv(annotations_file)

# ✅ Process and save images while keeping original filenames
for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Preprocessing Images"):
    img_path = os.path.join(data_dir, row['image_path'])  
    output_path = os.path.join(output_dir, os.path.basename(img_path))  # Keep the same filename

    if os.path.exists(output_path):  # Skip if already processed
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        resized_image = letterbox_image(image, (640, 480))
        resized_image.save(output_path)
    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")
