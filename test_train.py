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
from models import resnet18, resnet50, mobilenetv3, efficientnetb0, custom_cnn

MODEL_MAP = {
    "resnet18": resnet18.build_model,
    "resnet50": resnet50.build_model,
    "mobilenetv3": mobilenetv3.build_model,
    "efficientnetb0": efficientnetb0.build_model,
    "custom_cnn": custom_cnn.build_model,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train license plate detector")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--data-dir", required=True, help="Root directory of dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: only run a few batches and one epoch")
    parser.add_argument("--test-batches", type=int, default=5,
                        help="Number of training batches for dry run")
    parser.add_argument("--test-val-batches", type=int, default=5,
                        help="Number of validation batches for dry run")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda in use" if device.type == "cuda" else "CPU in use")

    # Determine save paths
    save_dir = os.path.join("weights", args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "metrics.csv")

    # Setup epochs
    if args.dry_run:
        num_epochs = 1
        print(f"Dry run mode: 1 epoch, up to {args.test_batches} train batches, "
              f"{args.test_val_batches} val batches.")
    else:
        num_epochs = 10

    # Initialize model
    model = MODEL_MAP[args.model](pretrained=True).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Prepare resume logic: load last best weights and set start_epoch
    start_epoch = 0
    if os.path.exists(log_file):
        # read last epoch from CSV
        with open(log_file) as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                start_epoch = max(start_epoch, int(rows[-1][0]))
    
    # read latest weight file
    model_files = glob.glob(os.path.join(save_dir, "best_model_epoch_*.pth"))
    last_weight_epoch = 0
    for mf in model_files:
        m = re.search(r"best_model_epoch_(\d+)\.pth", os.path.basename(mf))
        if m:
            last_weight_epoch = max(last_weight_epoch, int(m.group(1)))
    if last_weight_epoch > 0:
        weight_path = os.path.join(save_dir, f"best_model_epoch_{last_weight_epoch}.pth")
        print(f"Loading best model weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path))
        start_epoch = max(start_epoch, last_weight_epoch)

    # Data loaders
    train_dataset = LicensePlateIterableDataset(args.data_dir, os.path.join(args.data_dir, "combined_dataset/train_annotations.csv"))
    val_dataset = LicensePlateIterableDataset(args.data_dir, os.path.join(args.data_dir, "combined_dataset/val_annotations.csv"))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=12, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=12, pin_memory=True, persistent_workers=False)

    # Metrics & optimization
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Model params: {param_count:,}, size: {model_size_mb:.2f} MB")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val_loss = float("inf")

    # Ensure CSV header
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "IoU", "mAP",
                "precision", "recall", "F1_score", "FPS",
                "params", "model_size_mb",
            ])

    # Training loop with resume support
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}...")
        model.train()
        running_loss = 0.0
        iou_scores = []
        start_time = time.time()
        image_counter = 0

        # Training batches
        for batch_idx, (images, targets) in enumerate(
                tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")):
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
                iou_scores.append(
                    compute_iou(outputs[i].detach().cpu().numpy(),
                                targets[i].cpu().numpy())
                )
            if args.dry_run and batch_idx + 1 >= args.test_batches:
                break

        scheduler.step()
        train_loss = running_loss / (batch_idx + 1)
        fps = image_counter / (time.time() - start_time)
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou_scores = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(
                    tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", unit="batch")):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                for i in range(len(outputs)):
                    val_iou_scores.append(
                        compute_iou(outputs[i].cpu().numpy(),
                                    targets[i].cpu().numpy())
                    )
                if args.dry_run and batch_idx + 1 >= args.test_val_batches:
                    break

        val_loss /= (batch_idx + 1)
        mean_val_iou = sum(val_iou_scores) / len(val_iou_scores) if val_iou_scores else 0
        mAP, precision, recall, f1_score = calculate_metrics(val_iou_scores)

        print(
            f"[Epoch {epoch+1}] val_loss: {val_loss:.4f}, "
            f"IoU: {mean_val_iou:.4f}, Prec: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1_score:.4f}"
        )

        # Log metrics
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_loss, val_loss,
                mean_val_iou, mAP, precision,
                recall, f1_score, fps,
                param_count, model_size_mb,
            ])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Saved new best model: {best_model_path}")

    print("Training complete")

if __name__ == "__main__":
    main()
