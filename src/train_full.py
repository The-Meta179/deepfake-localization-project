
"""
Fully-supervised training for LAV-DF.

Training signal:
    frame-level labels.

The same EfficientNet-B0 + BiLSTM architecture is used as the
weak MIL model. The difference is the supervision:

    frame logits -> frame-level BCE loss

Video-level labels are NOT used for the fully supervised loss.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoClipDataset
from models import EfficientNetBiLSTM


FULL_BEST_NAME = "full_supervised_best.pth"
FULL_LAST_NAME = "full_supervised_last.pth"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )

    parser.add_argument(
        "--ckpt_dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    os.makedirs(
        args.ckpt_dir,
        exist_ok=True,
    )

    labels_csv = os.path.join(
        args.data_dir,
        "labels.csv",
    )

    full_best_path = os.path.join(
        args.ckpt_dir,
        FULL_BEST_NAME,
    )

    full_last_path = os.path.join(
        args.ckpt_dir,
        FULL_LAST_NAME,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    train_ds = VideoClipDataset(
        labels_csv,
        split="train",
        mode="full",
    )

    dev_ds = VideoClipDataset(
        labels_csv,
        split="dev",
        mode="full",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = EfficientNetBiLSTM(
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    start_epoch = 0

    # ------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------

    if os.path.exists(full_last_path):

        print(
            "Existing full-supervised checkpoint found:"
        )
        print(full_last_path)

        checkpoint = torch.load(
            full_last_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_val_loss = checkpoint[
            "best_val_loss"
        ]

        print(
            "Resuming from epoch:",
            start_epoch,
        )

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        model.train()

        train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/"
                f"{args.epochs} - Train"
            ),
        )

        for (
            clips,
            frame_labels,
        ) in progress:

            clips = clips.to(
                device,
                non_blocking=True,
            )

            frame_labels = frame_labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            # Model output:
            # video_logits [B]
            # frame_logits [B,T]
            _, frame_logits = model(
                clips
            )

            # IMPORTANT:
            # Full supervision uses ONLY frame labels.
            loss = criterion(
                frame_logits,
                frame_labels,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress.set_postfix(
                loss=loss.item()
            )

        train_loss /= len(
            train_loader
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for (
                clips,
                frame_labels,
            ) in dev_loader:

                clips = clips.to(
                    device,
                    non_blocking=True,
                )

                frame_labels = frame_labels.to(
                    device,
                    non_blocking=True,
                )

                _, frame_logits = model(
                    clips
                )

                loss = criterion(
                    frame_logits,
                    frame_labels,
                )

                val_loss += loss.item()

        val_loss /= len(
            dev_loader
        )

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        # --------------------------------------------------------
        # SAVE LAST
        # --------------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "train_loss":
                    train_loss,
                "val_loss":
                    val_loss,
                "best_val_loss":
                    best_val_loss,
            },
            full_last_path,
        )

        # --------------------------------------------------------
        # SAVE BEST
        # --------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "val_loss":
                        val_loss,
                },
                full_best_path,
            )

            print(
                "✅ New best fully supervised "
                "model saved."
            )

    print("\n==============================")
    print("FULL SUPERVISED TRAINING COMPLETE")
    print("==============================")
    print(
        "Best checkpoint:",
        full_best_path,
    )
    print(
        "Last checkpoint:",
        full_last_path,
    )


if __name__ == "__main__":
    main()
