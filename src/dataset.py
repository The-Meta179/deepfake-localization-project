"""
PyTorch Dataset classes. Both the weak and full arms read the SAME labels.csv
and use the SAME frame sampling/transform pipeline — only the label returned
differs. Keeping this identical is what makes the two arms comparable.
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
N_FRAMES_PER_CLIP = 16  # fixed-length clip sampled uniformly from each video

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def uniform_sample_indices(n_available, n_target):
    """Uniformly pick n_target frame indices out of n_available (with repeats if too few)."""
    if n_available >= n_target:
        step = n_available / n_target
        return [int(i * step) for i in range(n_target)]
    else:
        # repeat frames if the video was shorter than N_FRAMES_PER_CLIP after subsampling
        return [i % n_available for i in range(n_target)]


class VideoClipDataset(Dataset):
    """
    mode="full": returns (clip_tensor [T,C,H,W], frame_labels [T])
    mode="weak": returns (clip_tensor [T,C,H,W], video_label [scalar])
    """

    def __init__(self, labels_csv, split, mode):
        assert mode in ("full", "weak")
        self.mode = mode
        df = pd.read_csv(labels_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.video_ids = self.df["video_id"].unique().tolist()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        vid_df = self.df[self.df["video_id"] == video_id].sort_values("frame_idx").reset_index(drop=True)

        n_available = len(vid_df)
        sample_idx = uniform_sample_indices(n_available, N_FRAMES_PER_CLIP)

        frames = []
        frame_labels = []
        for i in sample_idx:
            row = vid_df.iloc[i]
            img = Image.open(row["frame_path"]).convert("RGB")
            frames.append(TRANSFORM(img))
            frame_labels.append(int(row["frame_label"]))

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        frame_labels = torch.tensor(frame_labels, dtype=torch.float32)  # [T]
        video_label = torch.tensor(float(vid_df["video_label"].iloc[0]))  # scalar

        if self.mode == "full":
            return clip, frame_labels
        else:
            return clip, video_label
