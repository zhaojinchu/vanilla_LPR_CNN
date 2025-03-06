import os
import pandas as pd
import shutil
import cv2

# Define dataset paths
ccpd_images_dir = "CCPD2019/ccpd_base"
kaggle_images_dir = "unprocessed_dataset/images"
kaggle_labels_dir = "unprocessed_dataset/labels"

output_images_dir = "combined_dataset/images"
annotations_file = "combined_dataset/annotations.csv"

# Ensure output directory exists
os.makedirs(output_images_dir, exist_ok=True)

# List to store annotations
annotations = []

# Function to parse CCPD filenames and extract bounding boxes
def parse_ccpd_filename(filename):
    parts = filename.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    bbox_part = parts[2]  # Example: "352&516_448&547"
    bbox_values = bbox_part.replace("&", "_").split("_")
    if len(bbox_values) != 4:
        raise ValueError(f"Bounding box extraction failed: {bbox_values} in {filename}")
    return map(int, bbox_values)  # Convert to integers

# Function to convert YOLO bounding boxes to absolute pixel values
def convert_yolo_to_absolute(label_file, img_width, img_height):
    try:
        with open(label_file, "r") as f:
            lines = f.readlines()

        if not lines:
            print(f"Skipping empty annotation file: {label_file}")
            return None

        # Check if multiple bounding boxes exist and skip those images
        if len(lines) > 1:
            print(f"Skipping {label_file} as it contains multiple bounding boxes.")
            return None

        parts = lines[0].strip().split()
        if len(parts) != 5:
            print(f"Skipping malformed annotation file: {label_file}")
            return None

        _, x_center, y_center, width, height = map(float, parts)

        # Convert from YOLO format to absolute pixel values
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)

        return x_min, y_min, x_max, y_max
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return None

# Process Kaggle dataset
for split in ["train", "val", "test"]:
    image_folder = os.path.join(kaggle_images_dir, split)
    label_folder = os.path.join(kaggle_labels_dir, split)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            label_file = filename.replace(".jpg", ".txt")
            label_path = os.path.join(label_folder, label_file)
            image_path = os.path.join(image_folder, filename)

            # Get actual image dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {image_path} as it could not be loaded.")
                continue
            img_height, img_width = img.shape[:2]

            bbox = convert_yolo_to_absolute(label_path, img_width, img_height)
            if bbox:
                x_min, y_min, x_max, y_max = bbox

                # Move image to single dataset folder
                dst_path = os.path.join(output_images_dir, filename)
                shutil.copy(image_path, dst_path)

                # Append annotation
                annotations.append([dst_path, x_min, y_min, x_max, y_max])

# Convert CCPD dataset
for filename in os.listdir(ccpd_images_dir):
    if filename.endswith(".jpg"):
        x_min, y_min, x_max, y_max = parse_ccpd_filename(filename)
        
        # Move image to single dataset folder
        src_path = os.path.join(ccpd_images_dir, filename)
        dst_path = os.path.join(output_images_dir, filename)
        shutil.copy(src_path, dst_path)

        # Append to annotations list
        annotations.append([dst_path, x_min, y_min, x_max, y_max])

# Save annotations to CSV
df = pd.DataFrame(annotations, columns=["image_path", "x_min", "y_min", "x_max", "y_max"])
df.to_csv(annotations_file, index=False)

print("Datasets merged and converted to CSV format!")
