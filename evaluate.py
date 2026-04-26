"""Evaluation entry point for trained checkpoints.

Run from project root:
    python evaluate.py

This script supports baseline evaluation and missing-modality stress tests.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import random_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
import csv
from pathlib import Path
import tqdm


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci_low: float = 2.75,
    ci_high: float = 97.5,
    seed: int = 42,
) -> tuple:
    """
    Compute a bootstrap confidence interval for any scalar metric.

    Matches the paper's CI range of 2.75% – 97.5% (≈ 95% CI).

    Args:
        y_true  : Ground-truth class indices (1-D int array).
        y_pred  : Predicted class indices (1-D int array).
        metric_fn : Callable(y_true, y_pred) -> float.
        n_boot  : Number of bootstrap resamples (default 1000).
        ci_low  : Lower percentile (default 2.75  to match paper).
        ci_high : Upper percentile (default 97.5  to match paper).
        seed    : Random seed for reproducibility.

    Returns:
        (lower, upper) confidence bounds as floats.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, ci_low)), float(np.percentile(scores, ci_high))

# Local project imports
# - Dataset handles loading/preprocessing/label encoding
# - Model loads the architecture used during training
from utils.dataset import AudioVisualDataset
from models.fusion_model import AudioVisualFusionModel
import config

def evaluate(missing_modality=None, missing_rate=0.0, save_csv=False, csv_path=None):
    """
    Evaluate a checkpoint on the dataset and report key metrics.

    Metrics reported (aligned with Table 2 of the RAVER paper):
      - Macro-F1   : unweighted average F1 across all classes
      - Micro-F1   : globally computed F1 (= accuracy for balanced sets)
      - Sample-Precision : weighted precision (per-sample average, matching
                           the "Sample-Precision" column in the paper)
      - Average Brier Score : calibration quality

    Args:
        missing_modality (str | None):
            - None: normal evaluation
            - 'audio': randomly zero audio for a fraction of samples
            - 'video': randomly zero video for a fraction of samples
        missing_rate (float): Probability in [0, 1] for applying the modality drop.
        save_csv (bool): If True, append metric row to `csv_path`.
        csv_path (str | Path | None): Path for CSV output.

    Returns:
        dict: Computed metrics.
    """
    device = torch.device(config.DEVICE)
    
    # Load full dataset
    dataset = AudioVisualDataset(
        data_path=config.CREMA_D_PATH,
    )

    # Same split as training
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # IMPORTANT: same seed as train.py
    generator = torch.Generator().manual_seed(42)

    _, test_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    test_loader = DataLoader(
    test_dataset,
    batch_size=config.EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=(config.DEVICE == "cuda")
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
            preds = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # --- Calculate Metrics ---
    # Convert probabilities to binary predictions using threshold 0.5
    # Convert probabilities to class predictions (multi-class)
    pred_classes = np.argmax(all_preds, axis=1)

    # Handle label format (important)
    if len(all_labels.shape) > 1:
        true_classes = np.argmax(all_labels, axis=1)  # if one-hot
    else:
        true_classes = all_labels  # if already class indices

    # ---------- Point-estimate Metrics (aligned with paper Table 2) ----------
    num_classes = config.NUM_CLASSES
    true_classes_int = np.array(true_classes, dtype=int)

    macro_f1      = f1_score(true_classes_int, pred_classes, average='macro',    zero_division=0)
    micro_f1      = f1_score(true_classes_int, pred_classes, average='micro',    zero_division=0)
    sample_prec   = precision_score(true_classes_int, pred_classes, average='weighted', zero_division=0)

    # Brier score (calibration)
    one_hot_labels = np.eye(num_classes)[true_classes_int]
    brier_scores   = [brier_score_loss(one_hot_labels[:, i], all_preds[:, i]) for i in range(num_classes)]
    avg_brier_score = float(np.mean(brier_scores))

    # Per-class F1
    per_class_f1 = f1_score(true_classes_int, pred_classes, average=None, zero_division=0)

    # ---------- Bootstrap Confidence Intervals (paper: 2.75% – 97.5%) ----------
    print("Computing bootstrap CIs (1000 resamples) …", flush=True)
    macro_ci  = bootstrap_ci(true_classes_int, pred_classes,
                             lambda yt, yp: f1_score(yt, yp, average='macro',    zero_division=0))
    micro_ci  = bootstrap_ci(true_classes_int, pred_classes,
                             lambda yt, yp: f1_score(yt, yp, average='micro',    zero_division=0))
    sprec_ci  = bootstrap_ci(true_classes_int, pred_classes,
                             lambda yt, yp: precision_score(yt, yp, average='weighted', zero_division=0))

    def _fmt(val, lo, hi):
        """Format as 'val (lo, hi)' matching paper style."""
        return f"{val:.3f} ({lo:.3f}, {hi:.3f})"

    # ---------- Print results ----------
    label = (f"{missing_modality} missing @ {missing_rate*100:.0f}%"
             if missing_modality else "Full modalities")
    emotion_names = {v: k for k, v in config.EMOTION_MAP.items()}
    class_names   = [emotion_names[i] for i in range(num_classes)]

    print(f"\n{'='*60}")
    print(f" Evaluation Results")
    print(f"{'='*60}")
    print(f"  {'Metric':<28} {'Value':>8}   {'95% CI (2.75–97.5%)':<26}")
    print(f"  {'-'*60}")
    print(f"  {'Macro-F1':<28} {macro_f1:>8.4f}   ({macro_ci[0]:.3f}, {macro_ci[1]:.3f})")
    print(f"  {'Micro-F1':<28} {micro_f1:>8.4f}   ({micro_ci[0]:.3f}, {micro_ci[1]:.3f})")
    print(f"  {'Sample-Precision':<28} {sample_prec:>8.4f}   ({sprec_ci[0]:.3f}, {sprec_ci[1]:.3f})")
    print(f"  {'Average Brier Score':<28} {avg_brier_score:>8.4f}")
    print(f"{'='*60}")
    print("Per-class F1:")
    for i, fval in enumerate(per_class_f1):
        print(f"  {emotion_names.get(i, str(i)):<12}: {fval:.4f}")
    print("-" * 60)
    print(classification_report(true_classes_int, pred_classes, target_names=class_names, zero_division=0))

    # ---------- Optional CSV save ----------
    metrics = {
        "condition":            label,
        "macro_f1":             round(float(macro_f1),       4),
        "macro_f1_ci_low":      round(macro_ci[0],            3),
        "macro_f1_ci_high":     round(macro_ci[1],            3),
        "micro_f1":             round(float(micro_f1),       4),
        "micro_f1_ci_low":      round(micro_ci[0],            3),
        "micro_f1_ci_high":     round(micro_ci[1],            3),
        "sample_precision":     round(float(sample_prec),    4),
        "sample_prec_ci_low":   round(sprec_ci[0],            3),
        "sample_prec_ci_high":  round(sprec_ci[1],            3),
        "avg_brier_score":      round(avg_brier_score,        4),
    }
    if save_csv and csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"Metrics appended to: {csv_path}")

    return metrics

if __name__ == "__main__":
    from pathlib import Path

    # CSV to collect all evaluation results in one place
    results_csv = Path("output/evaluation_results.csv")

    print("\n========================================")
    print(" AUDIO-VISUAL EMOTION RECOGNITION EVAL")
    print(" Metrics: Macro-F1 | Micro-F1 | Sample-Precision | Brier")
    print("========================================")

    evaluate(save_csv=True, csv_path=results_csv)

    print("\n--- Missing Audio Modality (100%) ---")
    evaluate(missing_modality='audio', missing_rate=0.5,
             save_csv=True, csv_path=results_csv)

    print("\n--- Missing Video Modality (100%) ---")
    evaluate(missing_modality='video', missing_rate=0.5,
             save_csv=True, csv_path=results_csv)

    # print("\n--- Missing Audio Modality (100%) ---")
    # evaluate(missing_modality='audio', missing_rate=1.0,
    #          save_csv=True, csv_path=results_csv)

    # print("\n--- Missing Video Modality (100%) ---")
    # evaluate(missing_modality='video', missing_rate=1.0,
    #          save_csv=True, csv_path=results_csv)

    print(f"\n✅ All results saved to: {results_csv}")
