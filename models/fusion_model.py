"""Top-level multimodal fusion model.

Pipeline:
1) Encode audio with HuBERT encoder.
2) Encode video with Swin/ViT encoder.
3) Fuse features using Perceiver IO.
4) Pool + classify emotions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_model import AudioEncoder
from models.video_model import VideoEncoder
from models.perceiver import PerceiverIO
import config

class AudioVisualFusionModel(nn.Module):
    """
    Main model that fuses audio and visual features with Perceiver IO.
    """
    def __init__(self, device='cpu'):
        super(AudioVisualFusionModel, self).__init__()
        
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        
        self.perceiver = PerceiverIO(
            depth=2,
            dim=config.PROJECTION_DIM,
            queries_dim=config.PROJECTION_DIM,
            num_latents=config.NUM_LATENTS,
            latent_dim=config.LATENT_DIM
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.LATENT_DIM),
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LATENT_DIM // 2, config.NUM_CLASSES)
        )

        # Keep pretrained encoders frozen; train fusion head only.
        self.freeze_pretrained_encoders()
        
        self.to(device)

    def freeze_pretrained_encoders(self):
        """Freeze HuBERT and Swin/ViT encoder weights.

        This keeps training focused on fusion/classification layers, reducing
        memory and compute requirements.
        """
        # Freeze everything
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        for p in self.video_encoder.parameters():
            p.requires_grad = False

        # 🔥 Unfreeze LAST layers safely (NO ASSUMPTIONS)

        # Audio (last 2 layers only)
        for name, param in self.audio_encoder.named_parameters():
            if "layers.10" in name or "layers.11" in name:
                param.requires_grad = True

        # Video (last stage)
        for name, param in self.video_encoder.named_parameters():
            if "layers.3" in name:
                param.requires_grad = True

    def trainable_parameters(self):
        """Return only parameters that should be optimized.

        Returns:
            generator[torch.nn.Parameter]: parameters with `requires_grad=True`.
        """
        return (p for p in self.parameters() if p.requires_grad)

    def forward(self, audio, video):
        """
        Args:
            audio (torch.Tensor): Audio waveform.
            video (torch.Tensor): Video frames.
        """
        # --- Feature Extraction ---
        # Normalize video layout to (batch, frames, channels, height, width)
        # Accept both:
        # - channels-last:  (B, F, H, W, C)
        # - channels-first: (B, F, C, H, W)
        if video.dim() != 5:
            raise ValueError(
                f"Expected 5D video tensor, got shape: {tuple(video.shape)}"
            )

        if video.shape[2] == 3:
            # Already (B, F, C, H, W)
            pass
        elif video.shape[-1] == 3:
            # Convert from (B, F, H, W, C) -> (B, F, C, H, W)
            video = video.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(
                "Video tensor must be channels-first or channels-last with 3 channels. "
                f"Got shape: {tuple(video.shape)}"
            )

        if not self.training:
            with torch.no_grad():
                audio_features = self.audio_encoder(audio)
                video_features = self.video_encoder(video)
        else:
            audio_features = self.audio_encoder(audio)
            video_features = self.video_encoder(video)

        # ✅ pooling (OUTSIDE no_grad so it always runs)
        audio_features = audio_features.mean(dim=1)
        video_features = video_features.mean(dim=1)

        # Add sequence dim back
        audio_features = audio_features.unsqueeze(1)
        video_features = video_features.unsqueeze(1)
        
        # --- Concatenate Features ---
        # Concatenate along the sequence dimension
        fused_features = torch.cat((audio_features, video_features), dim=1)
        
        # --- Perceiver Fusion ---
        perceiver_output = self.perceiver(fused_features)
        
        # --- Pooling ---
        # Pool across the latent dimension
        pooled_output = self.pooling(perceiver_output.permute(0, 2, 1)).squeeze(-1)
        
        # --- Classification ---
        logits = self.classifier(pooled_output)
        
        return logits
