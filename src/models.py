
"""
Models for the completed LAV-DF deepfake localization experiment.

This architecture matches the model used to create the existing
weak_mil_best.pth and full_supervised_best.pth checkpoints.

Architecture:
    EfficientNet-B0
        -> BiLSTM
        -> video classifier
        -> frame classifier

Both supervision arms use the SAME architecture.
The difference is only in the training loss:
    weak MIL       -> video-level top-k supervision
    full supervised -> frame-level supervision
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetBiLSTM(nn.Module):
    """
    Shared EfficientNet-B0 + bidirectional LSTM model.

    Input:
        clips [B, T, C, H, W]

    Output:
        video_logits [B]
        frame_logits [B, T]
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        # EfficientNet-B0 used by the original trained checkpoints.
        self.cnn = efficientnet_b0(
            weights=None
        )

        # Remove ImageNet classification head.
        self.cnn.classifier = nn.Identity()

        cnn_out_dim = 1280

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = hidden_size * 2

        self.video_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                lstm_out_dim,
                1,
            ),
        )

        self.frame_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                lstm_out_dim,
                1,
            ),
        )

    def forward(self, clips):
        """
        Forward pass.

        clips:
            [B, T, C, H, W]

        Returns:
            video_logits:
                [B]

            frame_logits:
                [B, T]
        """

        B, T, C, H, W = clips.shape

        # Process every frame independently through EfficientNet.
        x = clips.view(
            B * T,
            C,
            H,
            W,
        )

        features = self.cnn(x)

        # Restore temporal dimension.
        features = features.view(
            B,
            T,
            -1,
        )

        # Temporal modeling.
        lstm_out, _ = self.lstm(features)

        # Frame-level localization logits.
        frame_logits = self.frame_classifier(
            lstm_out
        ).squeeze(-1)

        # Video-level classification uses the temporal mean
        # representation, matching the trained model interface.
        video_features = lstm_out.mean(
            dim=1
        )

        video_logits = self.video_classifier(
            video_features
        ).squeeze(-1)

        return video_logits, frame_logits
