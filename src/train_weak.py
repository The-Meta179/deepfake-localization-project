"""
Phase 2a: train the weakly-supervised (video-level labels only) model via MIL.

Loss is computed on the VIDEO-level pooled score only — frame-level labels are
never used here, by design. Frame-level scores are only produced for later
localization at inference time.

Auto-resumes from the latest checkpoint in --ckpt_dir if one exists.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoClipDataset
from models import WeaklySupervisedModel

CKPT_NAME = "weak_model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    labels_csv = os.path.join(args.data_dir, "labels.csv")
    ckpt_path = os.path.join(args.ckpt_dir, CKPT_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = VideoClipDataset(labels_csv, split="train", mode="weak")
    val_ds = VideoClipDataset(labels_csv, split="val", mode="weak")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = WeaklySupervisedModel(topk=args.topk).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for clip, video_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            clip, video_label = clip.to(device), video_label.to(device)
            optimizer.zero_grad()
            _, video_logit = model(clip)
            loss = criterion(video_logit, video_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clip, video_label in val_loader:
                clip, video_label = clip.to(device), video_label.to(device)
                _, video_logit = model(clip)
                val_loss += criterion(video_logit, video_label).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }, ckpt_path)

    print(f"Training complete. Final checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
