# %% [markdown]
# # Audio-Visual Emotion Recognition: Final Report
#
# ## Executive Summary
# - This notebook-style Python script generates a complete post-training report for the project.
# - It uses the trained checkpoint, evaluates performance, compares modality behavior, and exports visual/metric artifacts.
# - The report is focused on **audio + video** modalities (no text branch).
# - Outputs are saved in `output/final_report`.

# %%
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    brier_score_loss,
)

# Ensure project root imports work even when script is run from outside project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.dataset import AudioVisualDataset
from utils import preprocessing
from models.fusion_model import AudioVisualFusionModel


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Runtime/report settings
REPORT_MAX_SAMPLES = 1500   # keeps report fast on local machines
REPORT_BATCH_SIZE = 2
FRAME_LIMIT_FOR_INFERENCE = 8

# Resolve key paths relative to project root for reliable execution.
DATA_ROOT = Path(config.CREMA_D_PATH)
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT

CHECKPOINT_PATH = Path(config.CHECKPOINT_PATH)
if not CHECKPOINT_PATH.is_absolute():
    CHECKPOINT_PATH = PROJECT_ROOT / CHECKPOINT_PATH

# Output folders
OUTPUT_ROOT = PROJECT_ROOT / "output" / "final_report"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
PRED_DIR = OUTPUT_ROOT / "predictions"
for p in [OUTPUT_ROOT, FIG_DIR, TABLE_DIR, PRED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Resolved data root:", DATA_ROOT)
print("Resolved checkpoint path:", CHECKPOINT_PATH)
print("Output root:", OUTPUT_ROOT.resolve())


# %% [markdown]
# ## Dataset Overview
# - **Dataset source**: CREMA-D (paired audio/video files)
# - **Modalities used in this report**:
#   - Audio (WAV)
#   - Video (MP4/FLV frame sequences)
# - **Classes**: angry, happy, sad, neutral, fear, disgust

# %%
# Load dataset
full_dataset = AudioVisualDataset(data_path=str(DATA_ROOT), modality_dropout_rate=0.0)
print("Total paired samples:", len(full_dataset))

# Build metadata table from file list
rows: List[Dict[str, str]] = []
for pair in full_dataset.file_list:
    audio_path = pair["audio"]
    video_path = pair["video"]

    fname = os.path.basename(audio_path)
    key = fname.replace(".wav", "")
    parts = key.split("_")

    actor_id = parts[0] if len(parts) > 0 else "NA"
    statement = parts[1] if len(parts) > 1 else "NA"
    emotion_code = parts[2] if len(parts) > 2 else "NA"
    intensity = parts[3] if len(parts) > 3 else "NA"

    emotion = config.CREMA_D_LABELS.get(emotion_code, "unknown")

    rows.append(
        {
            "audio_key": key,
            "audio_path": audio_path,
            "video_path": video_path,
            "actor_id": actor_id,
            "statement": statement,
            "emotion_code": emotion_code,
            "intensity": intensity,
            "emotion": emotion,
        }
    )

df = pd.DataFrame(rows)
print("\nSample rows:")
print(df.head(5))

print("\nBasic stats:")
print(df.describe(include="all").T[["count"]].head(10))

print("\nModalities available:")
modalities_df = pd.DataFrame(
    [
        {"modality": "audio", "available": True},
        {"modality": "video", "available": True},
        {"modality": "image frames", "available": True},
    ]
)
print(modalities_df)

# Class distribution
class_counts = df["emotion"].value_counts().sort_index()
print("\nClass distribution:")
print(class_counts)

plt.figure(figsize=(8, 4))
class_counts.plot(kind="bar", color="teal")
plt.title("Class Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=160)
plt.show()


# %% [markdown]
# ## Data Preprocessing
#
# ### Audio preprocessing
# - Load with torchaudio (fallback to soundfile/wave if needed)
# - Resample to 16kHz
# - Convert to mono
# - Pad/trim to fixed length
#
# ### Video preprocessing
# - Read frames with OpenCV
# - Face detection with Haar Cascade
# - Resize to 224x224
# - Pad/trim to fixed frame count

# %%
# Demonstrate preprocessing on one sample
sample_row = df.iloc[0]
sample_audio_path = sample_row["audio_path"]
sample_video_path = sample_row["video_path"]

waveform = preprocessing.extract_audio_from_path(sample_audio_path)
frames = preprocessing.extract_frames_from_video(sample_video_path)

print("Audio path:", sample_audio_path)
print("Video path:", sample_video_path)
print("Waveform shape:", tuple(waveform.shape))
print("Frames shape:", tuple(frames.shape))

# Display one frame
idx = min(0, len(frames) - 1)
plt.figure(figsize=(4, 4))
plt.imshow(frames[idx])
plt.title("Preprocessed Video Frame")
plt.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "sample_preprocessed_frame.png", dpi=160)
plt.show()

# Display waveform
plt.figure(figsize=(10, 3))
plt.plot(waveform.squeeze(0).numpy())
plt.title("Preprocessed Audio Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(FIG_DIR / "sample_preprocessed_waveform.png", dpi=160)
plt.show()


# %% [markdown]
# ## Feature Engineering / Embeddings
# - **Audio embeddings**: extracted by frozen HuBERT encoder
# - **Video embeddings**: extracted by frozen Swin/ViT encoder
# - **Fusion features**: Perceiver IO latent representation

# %%
# Split for evaluation/reporting
all_idx = np.arange(len(df))
if REPORT_MAX_SAMPLES < len(all_idx):
    selected_idx = np.random.choice(all_idx, size=REPORT_MAX_SAMPLES, replace=False)
    selected_idx = np.sort(selected_idx)
else:
    selected_idx = all_idx

selected_df = df.iloc[selected_idx].reset_index(drop=True)
train_idx_local, val_idx_local = train_test_split(
    np.arange(len(selected_df)),
    test_size=0.2,
    random_state=SEED,
    stratify=selected_df["emotion"],
)

# Map local split back to original dataset indices
orig_train_idx = selected_idx[train_idx_local]
orig_val_idx = selected_idx[val_idx_local]

train_ds = Subset(full_dataset, orig_train_idx.tolist())
val_ds = Subset(full_dataset, orig_val_idx.tolist())

train_loader = DataLoader(train_ds, batch_size=REPORT_BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=REPORT_BATCH_SIZE, shuffle=False, num_workers=0)

print("Report train samples:", len(train_ds))
print("Report val samples:", len(val_ds))

# Save split files
selected_df.iloc[train_idx_local].to_csv(TABLE_DIR / "train_split.csv", index=False)
selected_df.iloc[val_idx_local].to_csv(TABLE_DIR / "val_split.csv", index=False)
print("Saved split tables to", TABLE_DIR)

# Build model and load trained checkpoint
device = torch.device(config.DEVICE)
model = AudioVisualFusionModel(device=device).to(device)
checkpoint_path = CHECKPOINT_PATH
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Loaded checkpoint:", checkpoint_path)

# Feature extraction demo batch
demo_batch = next(iter(val_loader))
audio_demo = demo_batch["audio"].to(device)
video_demo = demo_batch["video"].to(device)[:, :FRAME_LIMIT_FOR_INFERENCE]

with torch.no_grad():
    audio_features = model.audio_encoder(audio_demo)
    video_features = model.video_encoder(video_demo)

print("Audio feature shape:", tuple(audio_features.shape))
print("Video feature shape:", tuple(video_features.shape))


# %% [markdown]
# ## Model Training
# - The final model is trained via `train.py` using fusion-only optimization.
# - In this report script, we load the trained checkpoint and visualize available training history if present.

# %%
history_csv_candidates = [
    PROJECT_ROOT / "artifacts" / "sample_training" / "epoch_history.csv",
    PROJECT_ROOT / "artifacts" / "epoch_history.csv",
]

history_df = None
for cand in history_csv_candidates:
    if cand.exists():
        history_df = pd.read_csv(cand)
        print("Loaded training history:", cand)
        break

if history_df is not None and len(history_df) > 0:
    print(history_df.head(10))

    plt.figure(figsize=(10, 4))
    for exp in history_df["experiment"].unique():
        part = history_df[history_df["experiment"] == exp]
        plt.plot(part["epoch"], part["train_loss"], marker="o", label=f"{exp} train")
        plt.plot(part["epoch"], part["val_loss"], marker="x", linestyle="--", label=f"{exp} val")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves.png", dpi=160)
    plt.show()
else:
    print("No epoch history CSV found. Skipping training curve plot.")


# %% [markdown]
# ## Evaluation Metrics
# - Accuracy
# - Precision
# - Recall
# - F1-score
# - Classification report

# %%
def evaluate_mode(mode: str = "combined") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model under a selected modality setting.

    Args:
        mode: one of {"combined", "audio_only", "video_only"}

    Returns:
        y_true_idx, y_pred_idx, y_prob (N x C)
    """
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)[:, :FRAME_LIMIT_FOR_INFERENCE]
            labels = batch["label"].to(device)

            if mode == "audio_only":
                video = torch.zeros_like(video)
            elif mode == "video_only":
                audio = torch.zeros_like(audio)

            logits = model(audio, video)
            # Use softmax (not sigmoid) — this is a multi-class problem
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    # labels_np may be 1-D class indices or 2-D one-hot — handle both
    if labels_np.ndim == 2:
        y_true_idx = labels_np.argmax(axis=1)
    else:
        y_true_idx = labels_np.astype(int)   # already class indices

    y_pred_idx = probs_np.argmax(axis=1)
    return y_true_idx, y_pred_idx, probs_np


y_true, y_pred, y_prob = evaluate_mode(mode="combined")

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
# Paper Table 2 columns ─────────────────────────────────────────────────────
micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
sample_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
# Brier score
y_true_int = np.array(y_true, dtype=int)
y_prob_brier_onehot = np.eye(config.NUM_CLASSES)[y_true_int]
brier_scores = [
    brier_score_loss(y_prob_brier_onehot[:, i], y_prob[:, i])
    for i in range(config.NUM_CLASSES)
]
avg_brier_score = float(np.mean(brier_scores))
# ────────────────────────────────────────────────────────────────────────────

idx_to_emotion = {v: k for k, v in config.EMOTION_MAP.items()}
class_names = [idx_to_emotion[i] for i in range(config.NUM_CLASSES)]

print("\n========= Evaluation Metrics (paper Table 2 aligned) =========")
print(f"{'Accuracy':<25}: {accuracy:.4f}")
print(f"{'Macro-F1':<25}: {f1:.4f}")
print(f"{'Micro-F1':<25}: {micro_f1:.4f}")
print(f"{'Sample-Precision (wt.)':<25}: {sample_precision:.4f}")
print(f"{'Average Brier Score':<25}: {avg_brier_score:.4f}")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

metrics_df = pd.DataFrame(
    [
        {"metric": "accuracy", "value": float(accuracy)},
        {"metric": "precision_macro", "value": float(precision)},
        {"metric": "recall_macro", "value": float(recall)},
        {"metric": "f1_macro",  "value": float(f1)},
        {"metric": "f1_micro",  "value": float(micro_f1)},
        {"metric": "sample_precision_weighted", "value": float(sample_precision)},
        {"metric": "avg_brier_score", "value": avg_brier_score},
    ]
)
metrics_df.to_csv(TABLE_DIR / "overall_metrics.csv", index=False)
print("Saved overall metrics:", TABLE_DIR / "overall_metrics.csv")


# %%
# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Combined)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
plt.show()


# %%
# ROC / AUC (one-vs-rest)
y_true_onehot = np.zeros((len(y_true), config.NUM_CLASSES), dtype=int)
y_true_onehot[np.arange(len(y_true)), y_true] = 1

plt.figure(figsize=(8, 6))
auc_rows = []
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
    class_auc = auc(fpr, tpr)
    auc_rows.append({"class": cname, "auc": float(class_auc)})
    plt.plot(fpr, tpr, label=f"{cname} (AUC={class_auc:.3f})")

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curves.png", dpi=160)
plt.show()

try:
    macro_auc = roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
except Exception:
    macro_auc = float(np.mean([r["auc"] for r in auc_rows]))

print("Macro ROC-AUC:", round(float(macro_auc), 4))

auc_df = pd.DataFrame(auc_rows)
auc_df.loc[len(auc_df)] = {"class": "macro", "auc": float(macro_auc)}
auc_df.to_csv(TABLE_DIR / "roc_auc_scores.csv", index=False)
print("Saved ROC AUC scores:", TABLE_DIR / "roc_auc_scores.csv")


# %% [markdown]
# ## Modality-wise Analysis
# - Performance comparison for:
#   - Audio only
#   - Video only
#   - Combined audio-video model

# %%
_, y_pred_audio, _ = evaluate_mode(mode="audio_only")
_, y_pred_video, _ = evaluate_mode(mode="video_only")
_, y_pred_combined, _ = evaluate_mode(mode="combined")

audio_acc = accuracy_score(y_true, y_pred_audio)
video_acc = accuracy_score(y_true, y_pred_video)
combined_acc = accuracy_score(y_true, y_pred_combined)

modality_df = pd.DataFrame(
    [
        {"modality": "Audio only", "accuracy": float(audio_acc)},
        {"modality": "Video only", "accuracy": float(video_acc)},
        {"modality": "Combined", "accuracy": float(combined_acc)},
    ]
)

print("Modality performance:")
print(modality_df.round(4))

plt.figure(figsize=(6, 4))
sns.barplot(data=modality_df, x="modality", y="accuracy", palette="viridis")
plt.title("Modality-wise Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "modality_comparison.png", dpi=160)
plt.show()

modality_df.to_csv(TABLE_DIR / "modality_performance.csv", index=False)
print("Saved modality performance:", TABLE_DIR / "modality_performance.csv")


# %% [markdown]
# ## Sample Predictions
# - Shows real validation examples with:
#   - Audio path
#   - Video path
#   - Predicted label
#   - Actual label

# %%
num_examples = min(5, len(val_ds))
preview_rows = []

for i in range(num_examples):
    sample = val_ds[i]
    audio = sample["audio"].unsqueeze(0).to(device)
    video = sample["video"].unsqueeze(0).to(device)[:, :FRAME_LIMIT_FOR_INFERENCE]
    true_onehot = sample["label"].cpu().numpy()

    with torch.no_grad():
        probs = torch.sigmoid(model(audio, video)).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    true_idx = int(np.argmax(true_onehot))

    orig_idx = int(orig_val_idx[i])
    audio_path = full_dataset.file_list[orig_idx]["audio"]
    video_path = full_dataset.file_list[orig_idx]["video"]

    preview_rows.append(
        {
            "audio_path": audio_path,
            "video_path": video_path,
            "predicted": idx_to_emotion[pred_idx],
            "actual": idx_to_emotion[true_idx],
            "correct": bool(pred_idx == true_idx),
        }
    )

    # show first frame
    frame0 = sample["video"][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.figure(figsize=(3, 3))
    plt.imshow(frame0)
    plt.title(f"Sample {i+1}: pred={idx_to_emotion[pred_idx]} | true={idx_to_emotion[true_idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

pred_df = pd.DataFrame(preview_rows)
print(pred_df)
pred_df.to_csv(PRED_DIR / "sample_predictions.csv", index=False)
print("Saved sample predictions:", PRED_DIR / "sample_predictions.csv")


# %% [markdown]
# ## Comparison with Paper Table 2 (CREMA-D)
# - Computes bootstrap confidence intervals (2.75% – 97.5%) to match the paper.
# - Compares against all CREMA-D baselines: TLSTM, SFAV, MulT, AuxFormer, VAVL, RAVER.

# %%
# ── Bootstrap CI helper (mirrors evaluate.py) ─────────────────────────────
def _bootstrap_ci(y_t, y_p, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(y_t)
    scores = [metric_fn(y_t[rng.integers(0, n, n)], y_p[rng.integers(0, n, n)])
              for _ in range(n_boot)]
    return float(np.percentile(scores, 2.75)), float(np.percentile(scores, 97.5))

# Point estimates already computed above as y_true / y_pred / y_prob
y_t = np.array(y_true, dtype=int)
y_p = np.array(y_pred, dtype=int)

print("Computing bootstrap CIs for your model …", flush=True)
our_macro_f1   = f1_score(y_t, y_p, average="macro",    zero_division=0)
our_micro_f1   = f1_score(y_t, y_p, average="micro",    zero_division=0)
our_sample_prec = precision_score(y_t, y_p, average="weighted", zero_division=0)

macro_lo,  macro_hi  = _bootstrap_ci(y_t, y_p, lambda a,b: f1_score(a, b, average="macro",    zero_division=0))
micro_lo,  micro_hi  = _bootstrap_ci(y_t, y_p, lambda a,b: f1_score(a, b, average="micro",    zero_division=0))
sprec_lo,  sprec_hi  = _bootstrap_ci(y_t, y_p, lambda a,b: precision_score(a, b, average="weighted", zero_division=0))

def _cell(val, lo, hi):
    return f"{val:.3f} ({lo:.3f}, {hi:.3f})"

# ── Hardcoded paper Table 2 baselines (CREMA-D) ───────────────────────────
paper_baselines = [
    # (model,       macro_f1, macro_lo, macro_hi,  micro_f1, micro_lo, micro_hi,  sprec,  sprec_lo, sprec_hi)
    ("TLSTM",       0.710, 0.704, 0.716,   0.705, 0.699, 0.711,   0.705, 0.699, 0.711),
    ("SFAV",        0.731, 0.725, 0.737,   0.731, 0.725, 0.736,   0.728, 0.723, 0.734),
    ("MulT",        0.743, 0.738, 0.750,   0.743, 0.738, 0.749,   0.741, 0.736, 0.748),
    ("AuxFormer",   0.742, 0.737, 0.748,   0.742, 0.737, 0.748,   0.741, 0.734, 0.747),
    ("VAVL",        0.772, 0.767, 0.778,   0.770, 0.765, 0.775,   0.770, 0.765, 0.775),
    ("RAVER",       0.777, 0.771, 0.782,   0.772, 0.766, 0.777,   0.772, 0.766, 0.777),
]

# ── Build comparison DataFrame ────────────────────────────────────────────
comp_rows = []
for row in paper_baselines:
    m, mf1, mlo, mhi, mif, milo, mihi, sp, splo, sphi = row
    comp_rows.append({
        "Model":           m,
        "Macro-F1":        _cell(mf1, mlo, mhi),
        "Micro-F1":        _cell(mif, milo, mihi),
        "Sample-Precision": _cell(sp,  splo, sphi),
        "_macro":          mf1,   # for bar chart
        "_micro":          mif,
        "_sprec":          sp,
    })

# Add your model (last row, bold indicator via model name)
comp_rows.append({
    "Model":           "Ours",
    "Macro-F1":        _cell(our_macro_f1,    macro_lo,  macro_hi),
    "Micro-F1":        _cell(our_micro_f1,    micro_lo,  micro_hi),
    "Sample-Precision": _cell(our_sample_prec, sprec_lo,  sprec_hi),
    "_macro":          our_macro_f1,
    "_micro":          our_micro_f1,
    "_sprec":          our_sample_prec,
})

comp_df = pd.DataFrame(comp_rows)

# ── Print table ───────────────────────────────────────────────────────────
display_df = comp_df[["Model", "Macro-F1", "Micro-F1", "Sample-Precision"]]
print("\n" + "="*75)
print(" Table 2 Comparison — CREMA-D  (CI: 2.75% – 97.5%)")
print("="*75)
print(display_df.to_string(index=False))
print("="*75)

display_df.to_csv(TABLE_DIR / "table2_comparison.csv", index=False)
print("Saved comparison table:", TABLE_DIR / "table2_comparison.csv")

# ── Bar chart ─────────────────────────────────────────────────────────────
models   = comp_df["Model"].tolist()
x        = np.arange(len(models))
width    = 0.28
colors   = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width, comp_df["_macro"],  width, label="Macro-F1",        color=colors[0], alpha=0.85)
b2 = ax.bar(x,          comp_df["_micro"],  width, label="Micro-F1",        color=colors[1], alpha=0.85)
b3 = ax.bar(x + width,  comp_df["_sprec"],  width, label="Sample-Precision", color=colors[2], alpha=0.85)

# Highlight "Ours" bar group
for bar_group in [b1, b2, b3]:
    bar_group[-1].set_edgecolor("red")
    bar_group[-1].set_linewidth(2.0)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.60, 0.85)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("CREMA-D — Model Comparison (Table 2 replication)", fontsize=13)
ax.legend(fontsize=10)
ax.axhline(0.777, color="gray", linestyle="--", linewidth=0.8, label="RAVER Macro-F1")
plt.tight_layout()
plt.savefig(FIG_DIR / "table2_comparison.png", dpi=160)
plt.show()
print("Saved comparison chart:", FIG_DIR / "table2_comparison.png")


# %% [markdown]
# ## Final Conclusion
#
# ### Overall model performance
# - The script reports full evaluation metrics (accuracy, precision, recall, F1, ROC/AUC).
# - It visualizes confusion matrix and modality-wise performance.
# - **Comparison with paper baselines** (Table 2) is saved to `table2_comparison.csv` and `table2_comparison.png`.
#
# ### Best modality
# - Use `modality_performance.csv` and modality bar chart to identify the strongest setup.
#
# ### Key insights
# - Combined audio-video generally performs better than single-modality settings.
# - Fusion-only training remains computationally efficient while leveraging pretrained encoders.
#
# ### Possible improvements
# - Add fixed script-level train/val/test split persistence.
# - Add threshold calibration and class-wise probability calibration.
# - Add larger-scale training/evaluation on GPU for stronger results.

# %%
# Save report index
report_index = {
    "output_root": str(OUTPUT_ROOT.resolve()),
    "figures": [p.name for p in sorted(FIG_DIR.glob("*.png"))],
    "tables": [p.name for p in sorted(TABLE_DIR.glob("*.csv"))],
    "predictions": [p.name for p in sorted(PRED_DIR.glob("*.csv"))],
}

print("\nReport completed.")
print(pd.Series(report_index))

pd.Series(report_index).to_json(OUTPUT_ROOT / "report_index.json", indent=2)
print("Saved report index:", OUTPUT_ROOT / "report_index.json")
