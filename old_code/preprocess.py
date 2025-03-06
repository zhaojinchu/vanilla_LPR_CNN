import os
import pandas as pd
import shutil

# Define dataset paths
ccpd_images_dir = "CCPD2019/ccpd_base"
kaggle_images_dir = "images"
kaggle_labels_dir = "labels"

output_images_dir = "combined_dataset2/images"
annotations_file = "combined_dataset2/annotations.csv"

# Ensure output directory exists
os.makedirs(output_images_dir, exist_ok=True)

# List to store annotations
annotations = []

# Function to parse CCPD filenames and extract bounding boxes
def parse_ccpd_filename(filename):
    parts = filename.split("-")

    if len(parts) < 4:  # Ensure filename is correctly formatted
        raise ValueError(f"Unexpected filename format: {filename}")

    # Extract bounding box coordinates
    bbox_part = parts[2]  # Example: "352&516_448&547"
    bbox_values = bbox_part.replace("&", "_").split("_")

    if len(bbox_values) != 4:
        raise ValueError(f"Bounding box extraction failed: {bbox_values} in {filename}")

    x_min, y_min, x_max, y_max = map(int, bbox_values)

    return x_min, y_min, x_max, y_max

# Function to convert YOLO bounding boxes to absolute pixel values
def convert_yolo_to_absolute(label_file, img_width, img_height):
    try:
        with open(label_file, "r") as f:
            lines = f.readlines()

        if not lines:  # If the file is empty
            print(f"Skipping empty annotation file: {label_file}")
            return None

        # Use only the first bounding box (if multiple exist)
        parts = lines[0].strip().split()

        if len(parts) != 5:
            print(f"Skipping malformed annotation file: {label_file}")
            return None

        _, x_center, y_center, width, height = map(float, parts)

        # Convert from relative (YOLO format) to absolute pixel values
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

            # Assume all images are 640x480 (adjust if needed)
            img_width, img_height = 640, 480  

            bbox = convert_yolo_to_absolute(label_path, img_width, img_height)
            if bbox:
                x_min, y_min, x_max, y_max = bbox

                # Move image to dataset folder
                src_path = os.path.join(image_folder, filename)
                dst_path = os.path.join(output_images_dir, split, filename)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

                # Append annotation
                annotations.append([dst_path, x_min, y_min, x_max, y_max])

# Convert CCPD dataset
for filename in os.listdir(ccpd_images_dir):
    if filename.endswith(".jpg"):
        x_min, y_min, x_max, y_max = parse_ccpd_filename(filename)

        # Move image to new dataset folder
        src_path = os.path.join(ccpd_images_dir, filename)
        dst_path = os.path.join(output_images_dir, filename)
        shutil.copy(src_path, dst_path)

        # Append to annotations list
        annotations.append([dst_path, x_min, y_min, x_max, y_max])

# Save annotations to CSV
df = pd.DataFrame(annotations, columns=["image_path", "x_min", "y_min", "x_max", "y_max"])
df.to_csv(annotations_file, index=False)

print("Datasets merged and converted to CSV format!")
