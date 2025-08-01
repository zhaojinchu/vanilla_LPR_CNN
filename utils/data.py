import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image


def compute_iou(boxA, boxB):
    boxA = np.array(boxA, dtype=np.float32)
    boxB = np.array(boxB, dtype=np.float32)
    boxA = np.clip(boxA, 0, 640)
    boxB = np.clip(boxB, 0, 640)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxB_area = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    denom = float(boxA_area + boxB_area - inter_area)
    return inter_area / denom if denom > 0 else 0


def calculate_metrics(iou_scores, iou_threshold=0.5):
    tp = sum(iou >= iou_threshold for iou in iou_scores)
    fp = sum(iou < iou_threshold for iou in iou_scores)
    fn = 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_ap = np.mean(iou_scores) if len(iou_scores) > 0 else 0
    return mean_ap, precision, recall, f1_score


class LicensePlateIterableDataset(IterableDataset):
    def __init__(self, root_dir, annotation_file, transform=None, chunk_size=1024):
        self.root_dir = os.path.join(root_dir, "combined_dataset/preletterboxed")
        self.annotation_file = annotation_file
        self.transform = transform if transform else transforms.ToTensor()
        self.chunk_size = chunk_size

    def __iter__(self):
        annotations = pd.read_csv(self.annotation_file, chunksize=self.chunk_size)
        for chunk in annotations:
            for _, row in chunk.iterrows():
                img_path = os.path.join(self.root_dir, os.path.basename(row['image_path']))
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Skipping corrupted image: {img_path} ({e})")
                    continue
                x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)
                image_tensor = self.transform(image)
                bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
                yield image_tensor, bbox_tensor
