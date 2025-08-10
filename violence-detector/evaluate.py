import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from torch.utils.data import DataLoader
from models.violence_model import ViolenceClassifier
from datasets.violence_dataset import ViolenceTensorDataset

# --- Configuration ---
MODEL_PATH = "checkpoints/violence_model_best.pth"
DATA_DIR = "preprocessed_tensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "evaluation_results"

print(f"[INFO] Using device: {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Model ---
# Ensure you are importing the correct ViolenceClassifier from your models file
model = ViolenceClassifier(dropout_p=0.5) 
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)
print("[INFO] Model loaded successfully from:", MODEL_PATH)

# --- Load Validation Data ---
val_ds = ViolenceTensorDataset(os.path.join(DATA_DIR, "val"), transform=None)
val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=0)
print("[INFO] Validation data loaded.")

# --- Run Inference ---
print("[INFO] Running inference on validation data to collect predictions...")
all_labels = []
all_probs = []

with torch.no_grad():
    for clips, labels in tqdm(val_loader, desc="Validating"):
        clips = clips.to(DEVICE)
        
        logits = model(clips)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = (all_probs >= 0.5).astype(int)

# --- Generate and Save Reports ---

# 1. Classification Report (Text File)
print("\n[INFO] Generating Classification Report...")
report = classification_report(all_labels, all_preds, target_names=['NonFight', 'Fight'], digits=4)
report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)
print(report)
print(f"[SUCCESS] Saved classification_report.txt to '{OUTPUT_DIR}' folder.")

# 2. Confusion Matrix (Image File)
print("\n[INFO] Generating Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.style.use("ggplot")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NonFight', 'Fight'], yticklabels=['NonFight', 'Fight'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"[SUCCESS] Saved confusion_matrix.png to '{OUTPUT_DIR}' folder.")

# 3. ROC Curve (Image File)
print("\n[INFO] Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()
print(f"[SUCCESS] Saved roc_curve.png to '{OUTPUT_DIR}' folder.")

print("\n[COMPLETE] All evaluation reports have been generated.")