import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def adjust_bboxes(orig_w, orig_h, new_w, new_h, x_min, y_min, x_max, y_max):
    """
    Adjust bounding box coordinates from the original image to the letterboxed image.
    
    Args:
        orig_w (int): Original image width before preprocessing
        orig_h (int): Original image height before preprocessing
        new_w (int): Target width after letterboxing (default: 640)
        new_h (int): Target height after letterboxing (default: 480)
        x_min, y_min, x_max, y_max (int): Original bounding box coordinates
    
    Returns:
        (int, int, int, int): Adjusted bounding box for the letterboxed image
    """
    # Calculate scaling factor while maintaining aspect ratio
    scale = min(new_w / orig_w, new_h / orig_h)

    # Compute padding added during letterboxing
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)

    pad_x = (new_w - scaled_w) // 2  # Horizontal padding
    pad_y = (new_h - scaled_h) // 2  # Vertical padding

    # First, scale the bounding box coordinates
    x_min = int(x_min * scale)
    y_min = int(y_min * scale)
    x_max = int(x_max * scale)
    y_max = int(y_max * scale)

    # Then, apply padding based on the letterboxing direction
    if scaled_w < new_w:  # Letterboxing applied on left/right (height was limiting factor)
        x_min += pad_x
        x_max += pad_x
    elif scaled_h < new_h:  # Letterboxing applied on top/bottom (width was limiting factor)
        y_min += pad_y
        y_max += pad_y

    return x_min, y_min, x_max, y_max


# ✅ Paths
data_dir = "C:/Users/Zhaojin/OneDrive/Desktop/vanilla_LPR_CNN"
annotations_file = os.path.join(data_dir, "combined_dataset/annotations.csv")
output_dir = os.path.join(data_dir, "combined_dataset/preletterboxed2")  # Directory for resized images
output_annotations_file = os.path.join(data_dir, "combined_dataset/annotations_preletterboxed.csv")

os.makedirs(output_dir, exist_ok=True)

# ✅ Load dataset annotations
annotations = pd.read_csv(annotations_file)

# ✅ Create a new list to store adjusted annotations
new_annotations = []

# ✅ Process images and adjust bounding boxes
for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Processing Annotations"):
    img_path = os.path.join(data_dir, row['image_path'])  
    output_path = os.path.join(output_dir, os.path.basename(img_path))  # Keep the same filename

    try:
        # Open the original image to get its size
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size

        # Apply bounding box transformation
        x_min, y_min, x_max, y_max = adjust_bboxes(
            orig_w, orig_h, new_w=640, new_h=480,
            x_min=row['x_min'], y_min=row['y_min'], x_max=row['x_max'], y_max=row['y_max']
        )

        # Store new annotation
        new_annotations.append([os.path.join("combined_dataset/preletterboxed", os.path.basename(img_path)), x_min, y_min, x_max, y_max])

    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")

# ✅ Save the adjusted annotations as a new CSV file
new_annotations_df = pd.DataFrame(new_annotations, columns=['image_path', 'x_min', 'y_min', 'x_max', 'y_max'])
new_annotations_df.to_csv(output_annotations_file, index=False)

print(f"✅ Saved adjusted annotations to {output_annotations_file}")
