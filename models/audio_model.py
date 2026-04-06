"""Audio encoder module.

Uses a pretrained HuBERT backbone for representation learning and a projection
layer to align feature dimensionality with the fusion block.
"""

import torch
import torch.nn as nn
from transformers import HubertModel

import config

class AudioEncoder(nn.Module):
    """
    Audio encoder using pretrained HuBERT + projection.

    HuBERT parameters are frozen to keep training efficient.
    """
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.hubert = HubertModel.from_pretrained(
            config.AUDIO_MODEL_NAME,
            token=config.HUGGING_FACE_TOKEN
        )
        
        # Freeze the HuBERT model
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        # Projection layer to match the dimension
        self.projection = nn.Linear(self.hubert.config.hidden_size, config.PROJECTION_DIM)

    def forward(self, audio_input):
        """
        Args:
            audio_input (torch.Tensor): waveform shaped either
                - (B, num_samples) or
                - (B, 1, num_samples)

        Returns:
            torch.Tensor: Features of shape (B, seq_len, config.PROJECTION_DIM).
        """
        # Squeeze the channel dimension if it exists
        if audio_input.dim() == 3 and audio_input.shape[1] == 1:
            audio_input = audio_input.squeeze(1)
            
        outputs = self.hubert(audio_input)
        last_hidden_state = outputs.last_hidden_state
        
        # Project the features to the desired dimension
        projected_features = self.projection(last_hidden_state)
        
        return projected_features
