# evaluate_and_plot.py

import torch
import numpy as np
import pandas as pd
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os

# Import custom modules
try:
    from model import LipReadModel
    from dataset import get_dataloaders, char_to_int, int_to_char, VOCAB_SIZE
    from train import greedy_decoder, decode_targets
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure model.py, dataset.py, and train.py are in the same directory.")
    exit()

# --- Configuration ---
MANIFEST_PATH = 'grid_manifest.csv'
MODEL_PATH = 'saved_models/lipread_cnn_lstm_ctc_grid.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (MUST MATCH THE SAVED MODEL'S ARCHITECTURE)
BATCH_SIZE = 16
CNN_OUTPUT_DIM = 512
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
DROPOUT = 0.3

print(f"--- Starting Full Evaluation on Test Set ---")
print(f"Using device: {DEVICE}")

# --- 1. Load Model ---
model = LipReadModel(
    input_channels=1,
    cnn_output_dim=CNN_OUTPUT_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    lstm_layers=LSTM_LAYERS,
    num_classes=VOCAB_SIZE,
    dropout_p=DROPOUT
).to(DEVICE)

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# --- 2. Load Test Data ---
print("\n--- Loading Test Data ---")
try:
    _, _, test_loader = get_dataloaders(
        MANIFEST_PATH, char_to_int, BATCH_SIZE,
        test_split=0.1, val_split=0.1, num_workers=4
    )
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 3. Run Inference and Collect Predictions ---
print("\n--- Running Inference on Test Set ---")
all_preds = []
all_targets = []
flat_true_chars = []
flat_pred_chars = []

with torch.no_grad():
    for batch_data in tqdm(test_loader, desc="Evaluating"):
        videos, labels, input_lengths, label_lengths = batch_data
        if videos is None: continue
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)
        log_probs = model(videos)
        preds = greedy_decoder(log_probs, int_to_char)
        targets = decode_targets(labels, label_lengths, int_to_char)
        all_preds.extend(preds)
        all_targets.extend(targets)
        for true_sent, pred_sent in zip(targets, preds):
            min_len = min(len(true_sent), len(pred_sent))
            flat_true_chars.extend(list(true_sent[:min_len]))
            flat_pred_chars.extend(list(pred_sent[:min_len]))

# --- 4. Calculate and Print Accuracy Metrics ---
print("\n--- Final Accuracy Metrics ---")
cer = jiwer.cer(all_targets, all_preds)
wer = jiwer.wer(all_targets, all_preds)
print(f"Character Error Rate (CER): {cer:.4f}  (Character Accuracy: {1-cer:.2%})")
print(f"Word Error Rate (WER):      {wer:.4f}  (Word Accuracy: {1-wer:.2%})")

# --- 5. Generate and Save Confusion Matrix ---
print("\n--- Generating Confusion Matrix ---")
display_labels = list(" abcdefghijklmnopqrstuvwxyz")
cm = confusion_matrix(flat_true_chars, flat_pred_chars, labels=display_labels)
cm_sum = cm.sum(axis=1)[:, np.newaxis]
with np.errstate(divide='ignore', invalid='ignore'):
    cm_normalized = np.nan_to_num(cm.astype('float') / cm_sum)

cm_df = pd.DataFrame(cm_normalized, index=display_labels, columns=display_labels)
plt.figure(figsize=(20, 18))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(cm_df, annot=True, fmt='.1%', cmap='viridis', cbar_kws={'label': 'Prediction Frequency'})
plt.title('Normalized Confusion Matrix (True vs. Predicted Characters)', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.tight_layout()
output_filename_cm = 'confusion_matrix_final.png'
plt.savefig(output_filename_cm, dpi=150)
print(f"Confusion matrix saved to: {output_filename_cm}")
plt.close() # Close the plot to free up memory

# --- 6. Generate and Plot F1-Score Chart ---
print("\n--- Generating F1-Score Report and Plot ---")
report_text = classification_report(flat_true_chars, flat_pred_chars, labels=display_labels)
print("Classification Report (per character):")
print(report_text)
report_dict = classification_report(flat_true_chars, flat_pred_chars, labels=display_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg']).drop(columns=['support'])

plt.figure(figsize=(18, 8))
sns.barplot(x=report_df.index, y=report_df['f1-score'], palette='viridis')
plt.title('F1-Score per Character', fontsize=20)
plt.xlabel('Character', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
output_filename_f1 = 'f1_scores_final.png'
plt.savefig(output_filename_f1, dpi=150)
print(f"F1-Score plot saved to: {output_filename_f1}")
plt.close()

print("\n--- Evaluation Script Finished ---")