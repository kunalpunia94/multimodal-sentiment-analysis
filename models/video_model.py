"""Video encoder module.

Uses pretrained Swin/ViT per-frame feature extraction and projects outputs to
the fusion dimension.
"""

import torch
import torch.nn as nn
from transformers import SwinModel, ViTModel

import config

class VideoEncoder(nn.Module):
    """
    Video encoder using pretrained Swin Transformer or ViT.

    Backbone parameters are frozen; only downstream layers are trained.
    """
    def __init__(self):
        super(VideoEncoder, self).__init__()
        
        if 'swin' in config.VIDEO_MODEL_NAME:
            self.video_model = SwinModel.from_pretrained(
                config.VIDEO_MODEL_NAME,
                token=config.HUGGING_FACE_TOKEN
            )
        elif 'vit' in config.VIDEO_MODEL_NAME:
            self.video_model = ViTModel.from_pretrained(
                config.VIDEO_MODEL_NAME,
                token=config.HUGGING_FACE_TOKEN
            )
        else:
            raise ValueError(f"Unsupported video model: {config.VIDEO_MODEL_NAME}")

        # Freeze the video model
        for param in self.video_model.parameters():
            param.requires_grad = False
            
        # Projection layer to match the dimension
        self.projection = nn.Linear(self.video_model.config.hidden_size, config.PROJECTION_DIM)

    def forward(self, video_frames):
        """
        Args:
            video_frames (torch.Tensor): A tensor of video frames of shape 
                                         (batch_size, num_frames, 3, height, width).

        Returns:
            torch.Tensor: Features of shape (B, F, config.PROJECTION_DIM).
        """
        batch_size, num_frames, c, h, w = video_frames.shape
        
        # Reshape to (batch_size * num_frames, c, h, w) to process all frames at once
        # Use reshape to handle non-contiguous tensors safely.
        video_frames = video_frames.reshape(batch_size * num_frames, c, h, w)
        
        # The Swin transformer expects pixel_values to be of shape (batch_size, num_channels, height, width)
        outputs = self.video_model(pixel_values=video_frames)
        last_hidden_state = outputs.last_hidden_state
        
        # Pool the features from the last layer
        pooled_features = last_hidden_state.mean(dim=1)
        
        # Reshape back to (batch_size, num_frames, features)
        pooled_features = pooled_features.reshape(batch_size, num_frames, -1)
        
        # Project the features
        projected_features = self.projection(pooled_features)
        
        return projected_features
