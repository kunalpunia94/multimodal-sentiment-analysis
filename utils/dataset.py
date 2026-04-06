"""Dataset utilities for pairing audio and video files.

This module defines a PyTorch `Dataset` that:
1) Discovers matching audio/video files by filename stem.
2) Applies preprocessing for each modality.
3) Produces tensors ready for model input.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

from utils import preprocessing
import config

class AudioVisualDataset(Dataset):
    """
        PyTorch dataset for paired audio-visual emotion samples.

        Expected directory structure under `data_path`:
            - AudioWAV/*.wav
            - VideoFlash/*.mp4 (preferred) or *.flv

        Files are paired using the shared filename stem.
    """

    def __init__(self, data_path, modality_dropout_rate=0.0):
        """
        Args:
            data_path (str): Dataset root containing AudioWAV and VideoFlash folders.
            modality_dropout_rate (float): Probability of zeroing each modality
                independently during training-style augmentation.
        """
        self.data_path = data_path
        self.modality_dropout_rate = modality_dropout_rate
        
        self.audio_path = os.path.join(data_path, "AudioWAV")
        self.video_path = os.path.join(data_path, "VideoFlash")
        
        if not os.path.exists(self.audio_path) or not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Dataset folders not found. Please check the paths: "
                f"{self.audio_path}, {self.video_path}"
            )
            
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        """Build list of valid audio/video file pairs.

        Returns:
            list[dict[str, str]]: each item has keys `audio` and `video`.
        """
        audio_files = glob(os.path.join(self.audio_path, "*.wav"))
        
        file_map = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file).replace('.wav', '')
            # Prefer MP4 (current dataset), fallback to FLV for compatibility
            video_file_mp4 = os.path.join(self.video_path, filename + '.mp4')
            video_file_flv = os.path.join(self.video_path, filename + '.flv')

            if os.path.exists(video_file_mp4):
                file_map.append({'audio': audio_file, 'video': video_file_mp4})
            elif os.path.exists(video_file_flv):
                file_map.append({'audio': audio_file, 'video': video_file_flv})
        
        return file_map
            
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Return one processed training sample.

        Returns:
            dict with keys:
                - 'audio': Tensor shape (1, MAX_AUDIO_LEN)
                - 'video': Tensor shape (F, C, H, W)
                - 'label': Multi-hot Tensor shape (NUM_CLASSES,)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_pair = self.file_list[idx]
        audio_path = file_pair['audio']
        video_path = file_pair['video']
        
        # --- Preprocessing ---
        audio_waveform = preprocessing.extract_audio_from_path(audio_path)
        video_frames = preprocessing.extract_frames_from_video(video_path)
        
        # --- Label ---
        filename = os.path.basename(audio_path)
        label_str = preprocessing.get_label_from_filename(filename)
        label = preprocessing.convert_label_to_multi_label(label_str)
        
        # --- Modality Dropout ---
        # Applied per sample, per modality.
        if self.modality_dropout_rate > 0:
            if torch.rand(1) < self.modality_dropout_rate:
                audio_waveform = torch.zeros_like(audio_waveform)
            if torch.rand(1) < self.modality_dropout_rate:
                video_frames = np.zeros_like(video_frames)
        
        return {
            'audio': audio_waveform,
            'video': torch.from_numpy(video_frames).float().permute(0, 3, 1, 2), # (F, C, H, W)
            'label': label
        }
