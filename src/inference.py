"""
Phase 5 (demo): run a trained model on an ARBITRARY uploaded video and report
which time segments are predicted as manipulated.

Why sliding windows: both models were trained on fixed 16-frame clips. A
1-2 minute video has far more frames than that, so a single forward pass
can't cover the whole thing. This script slides a 16-frame window across the
full video (with overlap), collects a frame-level score for every frame from
every window that covered it, averages overlapping scores, then thresholds
to produce final predicted segments with real timestamps.

Usage:
  python inference.py --video path/to/video.mp4 --model_type full \
      --ckpt_dir <ckpt_dir> --max_duration_sec 120
"""
import argparse
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms

from models import FullySupervisedModel, WeaklySupervisedModel

IMG_SIZE = 224
WINDOW_SIZE = 16      # must match N_FRAMES_PER_CLIP used in training
STRIDE = 8            # 50% overlap between windows
SAMPLE_FPS = 2.0      # must match --fps used in data_prep.py
SCORE_THRESHOLD = 0.5

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_sampled_frames(video_path, sample_fps):
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration_sec = total_frames / native_fps

    step = max(1, round(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))

    frames = []
    timestamps = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        frames.append(frame)
        timestamps.append(idx / native_fps)

    return frames, timestamps, duration_sec


def frames_to_segments(binary_labels, timestamps):
    """Convert per-sampled-frame 0/1 predictions into (start_sec, end_sec) segments."""
    segments = []
    in_seg = False
    start_t = None
    for i, v in enumerate(binary_labels):
        if v == 1 and not in_seg:
            start_t, in_seg = timestamps[i], True
        elif v == 0 and in_seg:
            segments.append((start_t, timestamps[i - 1]))
            in_seg = False
    if in_seg:
        segments.append((start_t, timestamps[-1]))
    return segments


def sliding_window_scores(model, model_type, frames, device):
    """Returns a per-frame averaged fake-probability score array, same length as frames."""
    n = len(frames)
    score_sum = np.zeros(n, dtype=np.float64)
    score_count = np.zeros(n, dtype=np.float64)

    tensors = [TRANSFORM(Image.fromarray(f)) for f in frames]

    starts = list(range(0, max(1, n - WINDOW_SIZE + 1), STRIDE))
    if not starts or starts[-1] + WINDOW_SIZE < n:
        starts.append(max(0, n - WINDOW_SIZE))  # ensure the tail is covered

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(start + WINDOW_SIZE, n)
            window = tensors[start:end]
            if len(window) < WINDOW_SIZE:
                # pad short final window by repeating the last frame
                window = window + [window[-1]] * (WINDOW_SIZE - len(window))
            clip = torch.stack(window, dim=0).unsqueeze(0).to(device)  # [1, T, C, H, W]

            if model_type == "full":
                logits = model(clip)  # [1, T]
            else:
                logits, _ = model(clip)  # [1, T]

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [T]

            for j in range(end - start):
                score_sum[start + j] += probs[j]
                score_count[start + j] += 1

    score_count[score_count == 0] = 1  # safety
    return score_sum / score_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_type", choices=["full", "weak"], required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--max_duration_sec", type=float, default=120.0,
                         help="Reject videos longer than this (demo compute limit)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames, timestamps, duration_sec = load_sampled_frames(args.video, SAMPLE_FPS)

    if duration_sec > args.max_duration_sec:
        print(f"REJECTED: video is {duration_sec:.1f}s, limit is {args.max_duration_sec:.0f}s. "
              f"Trim the video or raise --max_duration_sec (slower on free-tier GPU).")
        return

    if args.model_type == "full":
        model = FullySupervisedModel().to(device)
        ckpt = torch.load(os.path.join(args.ckpt_dir, "full_model.pt"), map_location=device)
    else:
        model = WeaklySupervisedModel().to(device)
        ckpt = torch.load(os.path.join(args.ckpt_dir, "weak_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print(f"Running {args.model_type} model on {args.video} "
          f"({duration_sec:.1f}s, {len(frames)} sampled frames)...")

    scores = sliding_window_scores(model, args.model_type, frames, device)
    binary_preds = (scores >= SCORE_THRESHOLD).astype(int)
    segments = frames_to_segments(binary_preds, timestamps)

    print("\n--- Result ---")
    if not segments:
        print("No manipulation detected above threshold.")
    else:
        print("Predicted manipulated segments:")
        for start_t, end_t in segments:
            print(f"  {start_t:6.2f}s  -  {end_t:6.2f}s")

    print(f"\n(Model: {args.model_type}, threshold: {SCORE_THRESHOLD}, "
          f"window: {WINDOW_SIZE} frames @ {SAMPLE_FPS} fps, stride: {STRIDE})")


if __name__ == "__main__":
    main()
