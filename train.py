"""Training entry point for audio-visual emotion recognition.

Run this file from the project root:
    python train.py

What this script does:
1) Builds the dataset and dataloader.
2) Instantiates the fusion model.
3) Optimizes only trainable fusion parameters (encoders are frozen).
4) Saves the best checkpoint based on training loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import tqdm
import os

# Local project imports
# - AudioVisualDataset: file pairing + preprocessing + labels
# - AudioVisualFusionModel: HuBERT + Swin/ViT + Perceiver fusion model
from utils.dataset import AudioVisualDataset
from models.fusion_model import AudioVisualFusionModel
import config

def train():
    """Run end-to-end training and save the best model checkpoint.

    The function uses configuration values from `config.py` and expects dataset
    folders under `config.CREMA_D_PATH`:
      - AudioWAV/
      - VideoFlash/

    Returns:
        None
    """
    # --- Setup ---
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
        
    device = torch.device(config.DEVICE)
    print("Runtime config:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Num workers: {config.NUM_WORKERS}")
    print(f"  Max frames: {config.MAX_FRAMES}")
    print(f"  Max audio sec: {config.MAX_AUDIO_LEN // config.SAMPLE_RATE}")
    print(f"  Audio model: {config.AUDIO_MODEL_NAME}")
    print(f"  Video model: {config.VIDEO_MODEL_NAME}")
    print(f"  Projection dim: {config.PROJECTION_DIM}")
    print(f"  Latent dim: {config.LATENT_DIM}")
    print(f"  Num latents: {config.NUM_LATENTS}")
    print(f"  AMP enabled: {config.USE_AMP}")
    print(f"  Grad accum steps: {config.GRAD_ACCUM_STEPS}")
    
    # --- Dataset and DataLoader ---
    train_dataset = AudioVisualDataset(
        data_path=config.CREMA_D_PATH,
        modality_dropout_rate=config.MODALITY_DROPOUT_RATE,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    
    # --- Model, Optimizer, and Loss ---
    model = AudioVisualFusionModel(device=device)
    # Important: only train fusion/head parameters for efficiency and stability.
    trainable_params = list(model.trainable_parameters())
    optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda" and config.USE_AMP))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params (fusion only): {trainable_param_count:,}")
    
    # --- Training Loop ---
    best_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(progress_bar):
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=(config.DEVICE == "cuda" and config.USE_AMP)):
                logits = model(audio, video)
                loss = criterion(logits, labels)

            # Backward pass and optimization
            scaled_loss = loss / max(config.GRAD_ACCUM_STEPS, 1)
            scaler.scale(scaled_loss).backward()

            if (step + 1) % max(config.GRAD_ACCUM_STEPS, 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Handle last partial accumulation window
        if len(train_loader) % max(config.GRAD_ACCUM_STEPS, 1) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {epoch_loss:.4f}")
        
        # --- Save Best Model ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))
            print("Saved best model.")
            
    print("Training finished.")

if __name__ == "__main__":
    train()
