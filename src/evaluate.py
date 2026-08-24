
"""
Evaluate the existing trained weak-MIL and fully-supervised models
on the held-out LAV-DF test split.

Both models produce:
    video_logits [B]
    frame_logits [B,T]

Localization is evaluated from frame-level probabilities.

The weak model is evaluated against frame-level ground truth only
at TEST TIME. Those frame labels were not used during weak training.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from dataset import VideoClipDataset
from models import EfficientNetBiLSTM


# Annotation-cost comparison used by the original experiment.
FULLY_SUPERVISED_MIN_PER_VIDEO = 20.0
WEAKLY_SUPERVISED_SEC_PER_VIDEO = 30.0


def frames_to_segments(binary_labels):
    """
    Convert 1D binary frame predictions into consecutive segments.

    Returns:
        list of (start, end) frame-index tuples.
    """

    segments = []

    in_seg = False
    start = None

    for i, value in enumerate(binary_labels):

        if value == 1 and not in_seg:
            start = i
            in_seg = True

        elif value == 0 and in_seg:
            segments.append(
                (start, i - 1)
            )
            in_seg = False

    if in_seg:
        segments.append(
            (
                start,
                len(binary_labels) - 1,
            )
        )

    return segments


def iou(seg_a, seg_b):
    """
    Intersection-over-Union between two frame-index segments.
    """

    start = max(
        seg_a[0],
        seg_b[0],
    )

    end = min(
        seg_a[1],
        seg_b[1],
    )

    intersection = max(
        0,
        end - start + 1,
    )

    union = (
        seg_a[1] - seg_a[0] + 1
        +
        seg_b[1] - seg_b[0] + 1
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def segment_ap(
    pred_segments,
    gt_segments,
    iou_thresh=0.5,
):
    """
    Simple segment AP@IoU used by the original evaluation code.
    """

    if len(pred_segments) == 0:
        return 0.0

    matched = 0

    for pred in pred_segments:

        if any(
            iou(pred, gt) >= iou_thresh
            for gt in gt_segments
        ):
            matched += 1

    return matched / len(
        pred_segments
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):
    """
    Evaluate frame-level localization.

    Returns:
        precision
        recall
        f1
        ap@0.5
    """

    all_preds = []
    all_labels = []

    ap_scores = []

    model.eval()

    for clips, frame_labels in loader:

        clips = clips.to(
            device,
            non_blocking=True,
        )

        # EfficientNetBiLSTM returns:
        #   video_logits [B]
        #   frame_logits [B,T]
        _, frame_logits = model(
            clips
        )

        probabilities = torch.sigmoid(
            frame_logits
        ).cpu().numpy()

        labels = frame_labels.numpy()

        for b in range(
            probabilities.shape[0]
        ):

            pred_binary = (
                probabilities[b] >= 0.5
            ).astype(int)

            gt_binary = (
                labels[b]
                .astype(int)
            )

            all_preds.extend(
                pred_binary.tolist()
            )

            all_labels.extend(
                gt_binary.tolist()
            )

            pred_segments = frames_to_segments(
                pred_binary
            )

            gt_segments = frames_to_segments(
                gt_binary
            )

            if len(gt_segments) > 0:

                ap_scores.append(
                    segment_ap(
                        pred_segments,
                        gt_segments,
                    )
                )

    return {
        "precision": precision_score(
            all_labels,
            all_preds,
            zero_division=0,
        ),

        "recall": recall_score(
            all_labels,
            all_preds,
            zero_division=0,
        ),

        "f1": f1_score(
            all_labels,
            all_preds,
            zero_division=0,
        ),

        "ap@0.5": (
            float(
                np.mean(ap_scores)
            )
            if ap_scores
            else 0.0
        ),
    }


def load_model_checkpoint(
    checkpoint_path,
    device,
):
    """
    Construct the verified architecture and load one of the
    existing checkpoints.
    """

    model = EfficientNetBiLSTM(
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, checkpoint


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
        "--batch_size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    labels_csv = os.path.join(
        args.data_dir,
        "labels.csv",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("==============================")
    print("LAV-DF MODEL EVALUATION")
    print("==============================")
    print("Device:", device)

    # ------------------------------------------------------------
    # TEST DATA
    # ------------------------------------------------------------

    # Frame-level labels are needed for localization scoring.
    # They are NOT used to train the weak model.
    test_ds = VideoClipDataset(
        labels_csv,
        split="test",
        mode="full",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Test videos:",
        len(test_ds),
    )

    # ------------------------------------------------------------
    # LOAD FULL MODEL
    # ------------------------------------------------------------

    full_path = os.path.join(
        args.ckpt_dir,
        "full_supervised_best.pth",
    )

    full_model, full_ckpt = (
        load_model_checkpoint(
            full_path,
            device,
        )
    )

    print(
        "\nFull checkpoint epoch:",
        full_ckpt["epoch"],
    )

    # ------------------------------------------------------------
    # LOAD WEAK MODEL
    # ------------------------------------------------------------

    weak_path = os.path.join(
        args.ckpt_dir,
        "weak_mil_best.pth",
    )

    weak_model, weak_ckpt = (
        load_model_checkpoint(
            weak_path,
            device,
        )
    )

    print(
        "Weak checkpoint epoch:",
        weak_ckpt["epoch"],
    )

    # ------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------

    print(
        "\nEvaluating fully-supervised model..."
    )

    full_metrics = evaluate_model(
        full_model,
        test_loader,
        device,
    )

    print(
        "Full metrics:",
        full_metrics,
    )

    print(
        "\nEvaluating weakly-supervised model..."
    )

    weak_metrics = evaluate_model(
        weak_model,
        test_loader,
        device,
    )

    print(
        "Weak metrics:",
        weak_metrics,
    )

    # ------------------------------------------------------------
    # HEADLINE TRADE-OFF PLOT
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        [
            FULLY_SUPERVISED_MIN_PER_VIDEO
            * 60
        ],
        [
            full_metrics["f1"] * 100
        ],
        s=120,
        label="Fully Supervised",
    )

    ax.scatter(
        [
            WEAKLY_SUPERVISED_SEC_PER_VIDEO
        ],
        [
            weak_metrics["f1"] * 100
        ],
        s=120,
        label="Weakly Supervised",
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Annotation cost (seconds/video, log scale)"
    )

    ax.set_ylabel(
        "F1-score (%)"
    )

    ax.set_title(
        "Annotation Cost vs. Accuracy Trade-off"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(
        args.ckpt_dir,
        "tradeoff_plot.png",
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved trade-off plot:",
        out_path,
    )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n==============================")
    print("FINAL EVALUATION SUMMARY")
    print("==============================")

    print(
        f"Fully-supervised F1: "
        f"{full_metrics['f1'] * 100:.1f}%"
    )

    print(
        f"Fully-supervised AP@0.5: "
        f"{full_metrics['ap@0.5'] * 100:.1f}%"
    )

    print(
        f"Weakly-supervised F1: "
        f"{weak_metrics['f1'] * 100:.1f}%"
    )

    print(
        f"Weakly-supervised AP@0.5: "
        f"{weak_metrics['ap@0.5'] * 100:.1f}%"
    )

    accuracy_gap = (
        full_metrics["f1"]
        -
        weak_metrics["f1"]
    ) * 100

    print(
        f"F1 gap: {accuracy_gap:.1f} points"
    )

    cost_ratio = (
        FULLY_SUPERVISED_MIN_PER_VIDEO * 60
    ) / WEAKLY_SUPERVISED_SEC_PER_VIDEO

    print(
        f"Annotation cost ratio (full/weak): "
        f"{cost_ratio:.0f}x"
    )


if __name__ == "__main__":
    main()
