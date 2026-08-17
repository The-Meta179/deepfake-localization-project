"""
Phase 3: evaluate both trained models on the held-out test split and produce
the accuracy-vs-annotation-cost comparison plot (your headline figure).

Frame-level threshold of 0.5 is used to turn scores into predicted fake/real
frames for both arms, then segments are formed from consecutive fake frames
for IoU-based scoring.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from dataset import VideoClipDataset
from models import FullySupervisedModel, WeaklySupervisedModel

# From the survey (Fig. 2): use these to frame your annotation-cost comparison
FULLY_SUPERVISED_MIN_PER_VIDEO = 20.0
WEAKLY_SUPERVISED_SEC_PER_VIDEO = 30.0  # midpoint of the 20-40 sec range


def frames_to_segments(binary_labels):
    """Convert a 1D array of 0/1 frame predictions into a list of (start, end) index segments."""
    segments = []
    in_seg = False
    start = None
    for i, v in enumerate(binary_labels):
        if v == 1 and not in_seg:
            start, in_seg = i, True
        elif v == 0 and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(binary_labels) - 1))
    return segments


def iou(seg_a, seg_b):
    start = max(seg_a[0], seg_b[0])
    end = min(seg_a[1], seg_b[1])
    intersection = max(0, end - start + 1)
    union = (seg_a[1] - seg_a[0] + 1) + (seg_b[1] - seg_b[0] + 1) - intersection
    return intersection / union if union > 0 else 0.0


def segment_ap(pred_segments, gt_segments, iou_thresh=0.5):
    """Simple AP@IoU: fraction of predicted segments that match a ground-truth segment above threshold."""
    if len(pred_segments) == 0:
        return 0.0
    matched = 0
    for p in pred_segments:
        if any(iou(p, g) >= iou_thresh for g in gt_segments):
            matched += 1
    return matched / len(pred_segments)


@torch.no_grad()
def evaluate_full(model, loader, device):
    all_preds, all_labels = [], []
    ap_scores = []
    model.eval()
    for clip, frame_labels in loader:
        clip = clip.to(device)
        logits = model(clip)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = frame_labels.numpy()

        for b in range(probs.shape[0]):
            pred_binary = (probs[b] >= 0.5).astype(int)
            gt_binary = labels[b].astype(int)
            all_preds.extend(pred_binary.tolist())
            all_labels.extend(gt_binary.tolist())

            pred_segs = frames_to_segments(pred_binary)
            gt_segs = frames_to_segments(gt_binary)
            if len(gt_segs) > 0:
                ap_scores.append(segment_ap(pred_segs, gt_segs))

    return {
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "ap@0.5": float(np.mean(ap_scores)) if ap_scores else 0.0,
    }


@torch.no_grad()
def evaluate_weak(model, loader, device):
    all_preds, all_labels = [], []
    ap_scores = []
    model.eval()
    for clip, frame_labels in loader:
        # NOTE: pass mode="full" dataset here too, so we still have frame-level
        # ground truth to SCORE against, even though the model never trained on it.
        clip = clip.to(device)
        frame_scores, _ = model(clip)
        probs = torch.sigmoid(frame_scores).cpu().numpy()
        labels = frame_labels.numpy()

        for b in range(probs.shape[0]):
            pred_binary = (probs[b] >= 0.5).astype(int)
            gt_binary = labels[b].astype(int)
            all_preds.extend(pred_binary.tolist())
            all_labels.extend(gt_binary.tolist())

            pred_segs = frames_to_segments(pred_binary)
            gt_segs = frames_to_segments(gt_binary)
            if len(gt_segs) > 0:
                ap_scores.append(segment_ap(pred_segs, gt_segs))

    return {
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "ap@0.5": float(np.mean(ap_scores)) if ap_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    labels_csv = os.path.join(args.data_dir, "labels.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Both evaluated with frame-level ground truth available (mode="full" dataset)
    # — the weak model just never SAW it during training.
    test_ds = VideoClipDataset(labels_csv, split="test", mode="full")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    full_model = FullySupervisedModel().to(device)
    full_ckpt = torch.load(os.path.join(args.ckpt_dir, "full_model.pt"), map_location=device)
    full_model.load_state_dict(full_ckpt["model_state"])

    weak_model = WeaklySupervisedModel().to(device)
    weak_ckpt = torch.load(os.path.join(args.ckpt_dir, "weak_model.pt"), map_location=device)
    weak_model.load_state_dict(weak_ckpt["model_state"])

    print("Evaluating fully-supervised model...")
    full_metrics = evaluate_full(full_model, test_loader, device)
    print(full_metrics)

    print("Evaluating weakly-supervised model...")
    weak_metrics = evaluate_weak(weak_model, test_loader, device)
    print(weak_metrics)

    # --- Headline plot: accuracy vs annotation cost ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        [FULLY_SUPERVISED_MIN_PER_VIDEO * 60], [full_metrics["f1"] * 100],
        color="crimson", s=120, label="Fully Supervised"
    )
    ax.scatter(
        [WEAKLY_SUPERVISED_SEC_PER_VIDEO], [weak_metrics["f1"] * 100],
        color="steelblue", s=120, label="Weakly Supervised"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Annotation cost (seconds/video, log scale)")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("Annotation Cost vs. Accuracy Trade-off (this project)")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(args.ckpt_dir, "tradeoff_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved trade-off plot to {out_path}")

    print("\n--- Summary ---")
    print(f"Fully-supervised F1: {full_metrics['f1']*100:.1f}%  AP@0.5: {full_metrics['ap@0.5']*100:.1f}%")
    print(f"Weakly-supervised F1: {weak_metrics['f1']*100:.1f}%  AP@0.5: {weak_metrics['ap@0.5']*100:.1f}%")
    print(f"Accuracy gap: {(full_metrics['f1'] - weak_metrics['f1'])*100:.1f} points")
    cost_ratio = (FULLY_SUPERVISED_MIN_PER_VIDEO * 60) / WEAKLY_SUPERVISED_SEC_PER_VIDEO
    print(f"Annotation cost ratio (full/weak): {cost_ratio:.0f}x")


if __name__ == "__main__":
    main()
