import argparse
import glob
import os
import re
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import LicensePlateIterableDataset, compute_iou, calculate_metrics
from models import resnet18, resnet50, mobilenetv3, efficientnetb0


MODEL_MAP = {
    "resnet18": resnet18.build_model,
    "resnet50": resnet50.build_model,
    "mobilenetv3": mobilenetv3.build_model,
    "efficientnetb0": efficientnetb0.build_model,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train license plate detector")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--data-dir", required=True, help="Root directory of dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LicensePlateIterableDataset(
        args.data_dir,
        os.path.join(args.data_dir, "combined_dataset/train_annotations.csv"),
    )
    val_dataset = LicensePlateIterableDataset(
        args.data_dir,
        os.path.join(args.data_dir, "combined_dataset/val_annotations.csv"),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=True)

    model = MODEL_MAP[args.model](pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    num_epochs = 10
    save_dir = os.path.join("weights", args.model)
    os.makedirs(save_dir, exist_ok=True)

    model_files = glob.glob(os.path.join(save_dir, "best_model_epoch_*.pth"))
    start_epoch = 0
    best_val_loss = float("inf")

    if model_files:
        epochs_found = []
        for mf in model_files:
            match = re.search(r"best_model_epoch_(\d+)\.pth", os.path.basename(mf))
            if match:
                epochs_found.append(int(match.group(1)))

        if epochs_found:
            last_epoch_found = max(epochs_found)
            best_model_file = os.path.join(save_dir, f"best_model_epoch_{last_epoch_found}.pth")
            print(f"Loading best model weights from {best_model_file}")
            model.load_state_dict(torch.load(best_model_file))
            start_epoch = last_epoch_found
        else:
            print("No valid best_model_epoch_*.pth files found. Starting from scratch.")
    else:
        print("No existing best model file found. Starting from scratch.")

    log_file = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "mAP", "precision", "recall", "F1_score", "FPS"])
    else:
        df = csv.reader(open(log_file))
        rows = list(df)
        if len(rows) > 1:
            last_logged_epoch = int(rows[-1][0])
            start_epoch = max(start_epoch, last_logged_epoch)
        print(f"Resuming from epoch {start_epoch + 1}")

    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        iou_scores = []
        start_time = time.time()
        image_counter = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            image_counter += len(images)

            for i in range(len(outputs)):
                iou = compute_iou(outputs[i].detach().cpu().numpy(), targets[i].cpu().numpy())
                iou_scores.append(iou)

        scheduler.step()
        train_loss = running_loss / (batch_idx + 1)
        fps = image_counter / (time.time() - start_time)
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

        model.eval()
        val_loss = 0.0
        val_iou_scores = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", unit="batch")):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                for i in range(len(outputs)):
                    iou_val = compute_iou(outputs[i].cpu().numpy(), targets[i].cpu().numpy())
                    val_iou_scores.append(iou_val)

        val_loss /= (batch_idx + 1)
        mean_val_iou = sum(val_iou_scores) / len(val_iou_scores) if val_iou_scores else 0
        mAP, precision, recall, f1_score = calculate_metrics(val_iou_scores)

        print(
            f"[Epoch {epoch+1}] val_loss: {val_loss:.4f}, IoU: {mean_val_iou:.4f}, "
            f"Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                mean_val_iou,
                mAP,
                precision,
                recall,
                f1_score,
                fps,
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Saved new best model: {best_model_path}")

    print("Training complete")


if __name__ == "__main__":
    main()
