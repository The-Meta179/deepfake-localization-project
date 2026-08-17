"""
Shared frozen backbone + two heads. Keeping the backbone identical across both
supervision arms is deliberate — it isolates "supervision type" as the one
variable being compared, matching the survey's framing.
"""
import timm
import torch
import torch.nn as nn


class FrozenBackbone(nn.Module):
    """EfficientNet-B0, ImageNet-pretrained, frozen. Chosen for speed on free-tier GPUs."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.out_dim = self.backbone.num_features  # 1280 for efficientnet_b0

    def forward(self, x):
        # x: [B*T, C, H, W] -> [B*T, out_dim]
        return self.backbone(x)


class TemporalEncoder(nn.Module):
    """BiLSTM over per-frame backbone features, shared architecture for both arms."""

    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        # x: [B, T, in_dim] -> [B, T, out_dim]
        out, _ = self.lstm(x)
        return out


class FullySupervisedModel(nn.Module):
    """Per-frame binary classification head. Trained with frame-level labels."""

    def __init__(self):
        super().__init__()
        self.backbone = FrozenBackbone()
        self.temporal = TemporalEncoder(self.backbone.out_dim)
        self.classifier = nn.Linear(self.temporal.out_dim, 1)

    def forward(self, clip):
        # clip: [B, T, C, H, W]
        B, T, C, H, W = clip.shape
        feats = self.backbone(clip.view(B * T, C, H, W)).view(B, T, -1)
        temporal_feats = self.temporal(feats)
        frame_logits = self.classifier(temporal_feats).squeeze(-1)  # [B, T]
        return frame_logits


class WeaklySupervisedModel(nn.Module):
    """
    MIL head: produces per-frame scores (used for localization at inference),
    pooled via top-k mean into a single video-level score (used for training,
    since only video-level labels are available).
    """

    def __init__(self, topk=4):
        super().__init__()
        self.backbone = FrozenBackbone()
        self.temporal = TemporalEncoder(self.backbone.out_dim)
        self.frame_scorer = nn.Linear(self.temporal.out_dim, 1)
        self.topk = topk

    def forward(self, clip):
        B, T, C, H, W = clip.shape
        feats = self.backbone(clip.view(B * T, C, H, W)).view(B, T, -1)
        temporal_feats = self.temporal(feats)
        frame_scores = self.frame_scorer(temporal_feats).squeeze(-1)  # [B, T] (logits)

        k = min(self.topk, T)
        topk_scores, _ = torch.topk(frame_scores, k, dim=1)
        video_logit = topk_scores.mean(dim=1)  # [B]

        return frame_scores, video_logit
