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
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                        help="Model architecture to train")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory of dataset (with combined_dataset subfolder)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (increase to better utilize GPU)")
    parser.add_argument("--val-batch-size", type=int,
                        help="Validation batch size (defaults to batch size * 2)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Optimizer weight decay")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of DataLoader workers (increase to speed data loading)")
    parser.add_argument("--scheduler", choices=["cosine", "one_cycle"], default="cosine",
                        help="LR scheduler type")
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="Steps per epoch for OneCycleLR")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1 epoch, few batches")
    parser.add_argument("--test-batches", type=int, default=5,
                        help="Train batches for dry run")
    parser.add_argument("--test-val-batches", type=int, default=5,
                        help="Val batches for dry run")
    parser.add_argument("--train-steps-per-epoch", type=int, default=None,
                        help="Max train batches per epoch (shorter epoch) if set")
    parser.add_argument("--val-steps-per-epoch", type=int, default=None,
                        help="Max val batches per epoch if set")
    parser.add_argument("--verbose", action="store_true",
                        help="Print sample IoU values and LR" )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda in use" if device.type == "cuda" else "CPU in use")

    # Paths
    save_dir = os.path.join("weights", args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "metrics.csv")

    # Override for dry run
    if args.dry_run:
        args.epochs = 1
        print(f"Dry-run mode: 1 epoch, up to {args.test_batches} train batches, {args.test_val_batches} val batches")

    # Model
    model = MODEL_MAP[args.model](pretrained=True).to(device)
    if device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Resume logic
    start_epoch = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                start_epoch = int(rows[-1][0])
    model_files = glob.glob(os.path.join(save_dir, "best_model_epoch_*.pth"))
    if model_files:
        last_num, last_path = max(
            (int(re.search(r"best_model_epoch_(\d+)", os.path.basename(p)).group(1)), p)
            for p in model_files
        )
        if last_num > 0:
            print(f"Loading weights from {last_path}")
            model.load_state_dict(torch.load(last_path))
            start_epoch = max(start_epoch, last_num)

    # Data loaders
    train_dataset = LicensePlateIterableDataset(
        args.data_dir, os.path.join(args.data_dir, "combined_dataset/train_annotations.csv"))
    val_dataset = LicensePlateIterableDataset(
        args.data_dir, os.path.join(args.data_dir, "combined_dataset/val_annotations.csv"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=not isinstance(train_dataset, torch.utils.data.IterableDataset)
    )
    val_bs = args.val_batch_size or args.batch_size * 2
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False
    )

    # Metrics & optimizer
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Model params: {param_count:,}, size: {model_size_mb:.2f} MB")

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # LR scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-6
        )
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch
        )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val_loss = float("inf")

    # CSV header
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "IoU", "mAP",
                "precision", "recall", "F1_score", "FPS",
                "lr", "params", "model_size_mb"
            ])

    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch + args.epochs}")
        model.train()
        running_loss = 0.0
        iou_scores = []
        img_count = 0
        t0 = time.time()

        # Train
        for i, (imgs, targs) in enumerate(train_loader):
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outs = model(imgs)
                loss = criterion(outs, targs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            img_count += imgs.size(0)
            iou_scores += [
                compute_iou(outs[j].detach().cpu().numpy(), targs[j].cpu().numpy())
                for j in range(len(outs))
            ]
            # Early break for shorter epoch
            if args.dry_run and i+1 >= args.test_batches:
                break
            if args.train_steps_per_epoch and i+1 >= args.train_steps_per_epoch:
                break

        scheduler.step()
        train_loss = running_loss / (i+1)
        fps = img_count / (time.time() - t0)
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

        # Validate
        model.eval()
        val_loss = 0.0
        val_iou = []
        with torch.no_grad():
            for i, (imgs, targs) in enumerate(val_loader):
                imgs, targs = imgs.to(device), targs.to(device)
                outs = model(imgs)
                val_loss += criterion(outs, targs).item()
                val_iou += [
                    compute_iou(outs[j].detach().cpu().numpy(), targs[j].cpu().numpy())
                    for j in range(len(outs))
                ]
                # Early break for shorter epoch
                if args.dry_run and i+1 >= args.test_val_batches:
                    break
                if args.val_steps_per_epoch and i+1 >= args.val_steps_per_epoch:
                    break

        val_loss /= (i+1)
        mean_val_iou = sum(val_iou)/len(val_iou) if val_iou else 0
        mAP, prec, rec, f1 = calculate_metrics(val_iou)

        current_lr = optimizer.param_groups[0]['lr']
        if args.verbose:
            print(f"LR: {current_lr:.2e}")
            print("Sample IoUs:", val_iou[:10])

        print(
            f"[Epoch {epoch+1}] train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, IoU={mean_val_iou:.4f}, "
            f"Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, FPS={fps:.1f}"
        )

        # Log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, val_loss,
                mean_val_iou, mAP, prec, rec, f1,
                fps, current_lr, param_count, model_size_mb
            ])

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"✔ Saved best: {ckpt}")

    print("Training complete")

if __name__ == "__main__":
    main()
