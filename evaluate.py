"""Evaluation entry point for trained checkpoints.

Run from project root:
    python evaluate.py

This script supports baseline evaluation and missing-modality stress tests.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, brier_score_loss
import tqdm

# Local project imports
# - Dataset handles loading/preprocessing/label encoding
# - Model loads the architecture used during training
from utils.dataset import AudioVisualDataset
from models.fusion_model import AudioVisualFusionModel
import config

def evaluate(missing_modality=None, missing_rate=0.0):
    """
    Evaluate a checkpoint on the dataset and report key metrics.

    Args:
        missing_modality (str | None):
            - None: normal evaluation
            - 'audio': randomly zero audio for a fraction of samples
            - 'video': randomly zero video for a fraction of samples
        missing_rate (float): Probability in [0, 1] for applying the modality drop.

    Returns:
        None. Metrics are printed to stdout.
    """
    device = torch.device(config.DEVICE)
    
    # --- Dataset and DataLoader ---
    test_dataset = AudioVisualDataset(
        data_path=config.CREMA_D_PATH,  # Assuming test set is the same for now
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # --- Load Model ---
    model = AudioVisualFusionModel(device=device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loader, desc="Evaluating")
        for batch in progress_bar:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # --- Simulate Missing Modality ---
            if missing_modality == 'audio':
                if np.random.rand() < missing_rate:
                    audio = torch.zeros_like(audio)
            elif missing_modality == 'video':
                if np.random.rand() < missing_rate:
                    video = torch.zeros_like(video)

            logits = model(audio, video)
            preds = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # --- Calculate Metrics ---
    # Convert probabilities to binary predictions using threshold 0.5
    binary_preds = (all_preds > 0.5).astype(int)
    
    macro_f1 = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, binary_preds, average='micro', zero_division=0)
    
    # Brier score for each class
    brier_scores = [brier_score_loss(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    avg_brier_score = np.mean(brier_scores)
    
    print("\n--- Evaluation Results ---")
    if missing_modality:
        print(f"Missing Modality: {missing_modality} at {missing_rate*100}%")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Average Brier Score: {avg_brier_score:.4f}")
    print("------------------------")

if __name__ == "__main__":
    # Example of running different evaluation experiments
    print("--- Normal Condition ---")
    evaluate()
    
    print("\n--- Missing Audio Modality (50%) ---")
    evaluate(missing_modality='audio', missing_rate=0.5)
    
    print("\n--- Missing Video Modality (50%) ---")
    evaluate(missing_modality='video', missing_rate=0.5)
    
    print("\n--- Missing Audio Modality (100%) ---")
    evaluate(missing_modality='audio', missing_rate=1.0)
    
    print("\n--- Missing Video Modality (100%) ---")
    evaluate(missing_modality='video', missing_rate=1.0)
