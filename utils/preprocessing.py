"""Preprocessing helpers for audio-visual emotion recognition.

This module handles:
- Video loading, face detection, resizing, and fixed-length frame output.
- Audio loading with robust fallbacks, resampling, and fixed-length waveform output.
- Label extraction/encoding from CREMA-D style filenames.
"""

import os
import cv2
import numpy as np
import torchaudio
import torch
import warnings
import wave

try:
    import soundfile as sf
except Exception:
    sf = None

import config

# Lightweight OpenCV face detector (no TensorFlow dependency)
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def _is_lfs_pointer(file_path):
    """Check whether a file is a Git-LFS pointer instead of real media.

    Args:
        file_path (str): Path to candidate media file.

    Returns:
        bool: True if file header matches Git-LFS pointer signature.
    """
    try:
        with open(file_path, "rb") as f:
            head = f.read(256)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _detect_face_bbox(frame):
    """
    Detect the largest face in a frame using OpenCV Haar cascade.

    Returns:
        tuple[int, int, int, int] | None: (x, y, w, h) or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return None

    # Use the largest detected face for stability
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)

def extract_frames_from_video(video_path):
    """
    Extracts frames from a video file, detects faces, and resizes them.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.ndarray: Frame tensor with shape
            (config.MAX_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE, 3).
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return np.zeros((config.MAX_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)

    if _is_lfs_pointer(video_path):
        raise RuntimeError(
            f"Video file is a Git LFS pointer, not real media: {video_path}. "
            "Dataset files were not downloaded (LFS object missing)."
        )
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened() and len(frames) < config.MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = _detect_face_bbox(frame)
        if bbox is not None:
            x, y, w, h = bbox
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                resized_face = cv2.resize(face, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                frames.append(resized_face)
            else:
                resized_frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                frames.append(resized_frame)
        else:
            # Fallback if no face is detected
            resized_frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            frames.append(resized_frame)
    cap.release()
    
    if not frames:
        # If no faces are detected, return a black image sequence
        return np.zeros((config.MAX_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)

    frames = np.array(frames)
    
    # Pad or truncate frames to a fixed length
    num_frames = frames.shape[0]
    if num_frames < config.MAX_FRAMES:
        padding = np.zeros((config.MAX_FRAMES - num_frames, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        frames = np.concatenate([frames, padding], axis=0)
    
    return frames

def extract_audio_from_path(audio_path):
    """
    Extracts audio from an audio file and resamples it.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        torch.Tensor: Mono waveform of shape (1, config.MAX_AUDIO_LEN).
    """
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return torch.zeros((1, config.MAX_AUDIO_LEN))

    if _is_lfs_pointer(audio_path):
        raise RuntimeError(
            f"Audio file is a Git LFS pointer, not real media: {audio_path}. "
            "Dataset files were not downloaded (LFS object missing)."
        )

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        warnings.warn(
            f"torchaudio.load failed, trying fallbacks. Error: {e}",
            RuntimeWarning,
        )

        if sf is not None:
            data, sample_rate = sf.read(audio_path, always_2d=True)
            # soundfile returns (num_samples, num_channels)
            waveform = torch.from_numpy(data.T).float()
        else:
            # Final fallback: standard-library WAV reader (works for PCM WAV)
            with wave.open(audio_path, "rb") as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)

            dtype_map = {
                1: np.uint8,
                2: np.int16,
                4: np.int32,
            }
            if sample_width not in dtype_map:
                raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

            audio_np = np.frombuffer(raw, dtype=dtype_map[sample_width])
            audio_np = audio_np.reshape(-1, num_channels)

            if sample_width == 1:
                # 8-bit PCM WAV is unsigned
                audio_np = (audio_np.astype(np.float32) - 128.0) / 128.0
            elif sample_width == 2:
                audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                audio_np = audio_np.astype(np.float32) / 2147483648.0

            waveform = torch.from_numpy(audio_np.T)
    
    # Resample to project sample rate if needed.
    if sample_rate != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if source has multiple channels.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Pad or truncate to a fixed number of samples.
    if waveform.shape[1] > config.MAX_AUDIO_LEN:
        waveform = waveform[:, :config.MAX_AUDIO_LEN]
    else:
        padding = torch.zeros((1, config.MAX_AUDIO_LEN - waveform.shape[1]))
        waveform = torch.cat([waveform, padding], dim=1)
        
    return waveform

def get_label_from_filename(filename):
    """
    Extracts the emotion label from a CREMA-D filename.
    Example:
        '1001_IEO_HAP_XX.wav' -> 'HAP'
    """
    parts = filename.split('_')
    if len(parts) > 2:
        return parts[2]
    return None

def convert_label_to_multi_label(label):
    """Convert emotion code into model target tensor.

    Args:
        label (str | None): short code, e.g. 'ANG', 'HAP', ...

    Returns:
        torch.Tensor: multi-hot vector with shape (config.NUM_CLASSES,).
    """
    target = torch.zeros(config.NUM_CLASSES)
    if label in config.CREMA_D_LABELS:
        emotion = config.CREMA_D_LABELS[label]
        if emotion in config.EMOTION_MAP:
            target[config.EMOTION_MAP[emotion]] = 1.0
    return target
