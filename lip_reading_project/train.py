# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from tqdm import tqdm
import jiwer # For CER/WER calculation

# Import custom modules
from dataset import get_dataloaders, char_to_int, int_to_char, VOCAB_SIZE, CHARSET
from model import LipReadModel

# --- Configuration ---
MANIFEST_PATH = 'grid_manifest.csv' # Path to your manifest file
MODEL_SAVE_DIR = 'saved_models'     # Directory to save best model
MODEL_NAME = 'lipread_cnn_lstm_ctc_grid.pth' # Filename for the saved model
LOG_FILE = 'training_log_grid.txt'    # Log file name

# Hyperparameters (Adjust as needed, especially BATCH_SIZE for your hardware)
BATCH_SIZE = 8        # Reduce if you encounter GPU memory issues (e.g., 8, 4)
LEARNING_RATE = 1e-4   # Initial learning rate
EPOCHS = 50            # Maximum number of training epochs
CLIP_GRAD_NORM = 5.0   # Gradient clipping threshold to prevent exploding gradients
PATIENCE = 10          # Epochs to wait for validation improvement before stopping early

# Model parameters (Ensure these match the defaults or your modifications in model.py)
CNN_OUTPUT_DIM = 512
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
DROPOUT = 0.3

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Helper: Greedy CTC Decoder ---
# Converts frame-wise probabilities to text using CTC rules
def greedy_decoder(log_probs, int_to_char_map):
    """
    Decodes CTC output using greedy approach (best path).
    Args:
        log_probs (Tensor): Log probabilities from the model (T, B, N).
        int_to_char_map (dict): Mapping from integer indices to characters.
    Returns:
        list: List of decoded strings for the batch.
    """
    decoded_texts = []
    # Get the most likely character index at each time step for the whole batch
    best_path = torch.argmax(log_probs, dim=2) # Shape: (T, B)

    for i in range(best_path.size(1)): # Iterate through each sequence in the batch
        sequence_indices = best_path[:, i].cpu().numpy() # Get sequence for batch item i

        # --- CTC Decoding Logic ---
        # 1. Remove consecutive duplicates
        # 2. Remove blank tokens (index 0 in our case)
        decoded_sequence_indices = []
        last_char_index = -1 # Use -1 to ensure the first non-blank is always added
        blank_index = char_to_int.get(' ', 0) # Get blank index dynamically

        for char_index in sequence_indices:
            if char_index != blank_index: # If not blank
                if char_index != last_char_index: # If different from last character added
                    decoded_sequence_indices.append(char_index)
            last_char_index = char_index # Update last index seen (important!)

        # Convert indices back to characters
        decoded_text = "".join([int_to_char_map.get(c, '?') for c in decoded_sequence_indices])
        decoded_texts.append(decoded_text)
    return decoded_texts

# --- Helper: Target Decoder ---
# Converts padded target labels back to text
def decode_targets(labels, label_lengths, int_to_char_map):
    """
    Decodes padded target labels into strings.
    Args:
        labels (Tensor): Padded target labels (B, L_max).
        label_lengths (Tensor): Actual lengths of each label sequence (B,).
        int_to_char_map (dict): Mapping from integer indices to characters.
    Returns:
        list: List of decoded target strings for the batch.
    """
    decoded_targets = []
    for i in range(labels.size(0)): # Iterate through batch
        actual_len = label_lengths[i].item()
        target_indices = labels[i][:actual_len].cpu().numpy() # Get only the actual label indices
        target_text = "".join([int_to_char_map.get(c, '?') for c in target_indices])
        decoded_targets.append(target_text)
    return decoded_targets

# --- Logging Function ---
def log_message(message):
    """Prints message to console and appends to log file."""
    print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# --- Training Function for One Epoch ---
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch_num):
    model.train() # Set model to training mode
    total_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{EPOCHS} Train", leave=False, dynamic_ncols=True)

    for batch_idx, batch_data in enumerate(progress_bar):
        # Unpack batch data (handle potential None from collate_fn)
        videos, labels, input_lengths, label_lengths = batch_data
        if videos is None:
            log_message(f"Warning: Skipping empty batch {batch_idx} in training.")
            continue

        # Move data to the appropriate device
        videos = videos.to(device)           # (B, T_max, C, H, W)
        labels = labels.to(device)           # (B, L_max)
        input_lengths = input_lengths.to(device) # (B,) - Actual lengths BEFORE padding
        label_lengths = label_lengths.to(device) # (B,) - Actual lengths BEFORE padding

        optimizer.zero_grad() # Reset gradients

        # --- Forward Pass ---
        # Model output should be log_probs with shape (T, B, N)
        log_probs = model(videos)
        T = log_probs.size(0) # Get time dimension from model output

        # --- Prepare for CTC Loss ---
        # Ensure input lengths do not exceed the output time dimension from the model
        input_lengths_for_loss = torch.clamp(input_lengths.long(), max=T)

        # Check for sequences with zero length after clamping (could happen with very short videos/errors)
        # Also ensure target lengths are > 0
        valid_indices_mask = (input_lengths_for_loss > 0) & (label_lengths.long() > 0)
        if not valid_indices_mask.all():
            # Filter batch or skip (skipping is simpler here)
            num_skipped = videos.size(0) - valid_indices_mask.sum().item()
            log_message(f"Warning: Skipping batch {batch_idx} due to {num_skipped} zero length inputs/labels after clamping.")
            continue # Skip this batch if any sequence has zero length

        # --- Calculate CTC Loss ---
        # Input shapes for nn.CTCLoss:
        # log_probs: (T, B, N) = (Time, Batch, Classes)
        # labels: (B, L) = (Batch, TargetLength)
        # input_lengths: (B,) = Length of each sequence in log_probs (must be <= T)
        # label_lengths: (B,) = Length of each target sequence in labels (must be <= L)
        loss = loss_fn(log_probs, labels, input_lengths_for_loss, label_lengths.long())

        # Check for invalid loss values
        if torch.isnan(loss) or torch.isinf(loss):
            log_message(f"Warning: Invalid loss encountered (NaN or Inf) in batch {batch_idx}. Skipping backward pass for this batch.")
            continue # Skip optimization step for this batch

        # --- Backward Pass & Optimization ---
        loss.backward()

        # Gradient Clipping (important for LSTMs to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

        optimizer.step() # Update model weights

        # --- Accumulate Loss and Update Progress Bar ---
        current_loss = loss.item()
        total_loss += current_loss
        progress_bar.set_postfix(loss=f"{current_loss:.4f}")

    # --- Return Average Loss for the Epoch ---
    avg_epoch_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return avg_epoch_loss

# --- Evaluation Function (for Validation and Test) ---
def evaluate(model, loader, loss_fn, device, int_to_char_map, epoch_num):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{EPOCHS} Val  ", leave=False, dynamic_ncols=True)

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch_idx, batch_data in enumerate(progress_bar):
            videos, labels, input_lengths, label_lengths = batch_data
            if videos is None:
                log_message(f"Warning: Skipping empty batch {batch_idx} in evaluation.")
                continue

            videos = videos.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            # --- Forward Pass ---
            log_probs = model(videos) # (T, B, N)
            T = log_probs.size(0)
            input_lengths_for_loss = torch.clamp(input_lengths.long(), max=T)

            # --- Check for zero lengths ---
            valid_indices_mask = (input_lengths_for_loss > 0) & (label_lengths.long() > 0)
            if not valid_indices_mask.all():
                 num_skipped = videos.size(0) - valid_indices_mask.sum().item()
                 # Optionally log skipping, but might be too verbose for validation
                 continue # Skip this batch

            # --- Calculate Loss (Optional, but good for monitoring) ---
            loss = loss_fn(log_probs, labels, input_lengths_for_loss, label_lengths.long())
            # Skip if loss is invalid
            if not (torch.isnan(loss) or torch.isinf(loss)):
                 total_loss += loss.item()
            else:
                 log_message(f"Warning: Invalid loss encountered during evaluation batch {batch_idx}.")


            # --- Decode Predictions and Targets ---
            preds = greedy_decoder(log_probs, int_to_char_map)
            targets = decode_targets(labels, label_lengths, int_to_char_map)
            all_preds.extend(preds)
            all_targets.extend(targets)

    # --- Calculate Average Loss and CER ---
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    # Calculate Character Error Rate (CER) - Ensure lists are not empty
    cer = float('inf') # Default to infinity if no valid evaluation happened
    if all_targets and all_preds:
        try:
            cer = jiwer.cer(all_targets, all_preds)
        except Exception as e:
            log_message(f"Error calculating CER with jiwer: {e}")


    # --- Print Some Examples ---
    log_message("\n--- Validation Examples ---")
    num_examples = min(5, len(all_preds))
    for i in range(num_examples):
        log_message(f" Target: '{all_targets[i]}'")
        log_message(f" Pred:   '{all_preds[i]}'")
        log_message("-" * 20)

    return avg_loss, cer

# --- Main Execution Block ---
if __name__ == '__main__':
    # Clear log file at the start of training
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log_message("--- Lip Reading Model Training Started ---")
    log_message(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Using Device: {DEVICE}")
    log_message(f"Configuration: Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Epochs={EPOCHS}, Patience={PATIENCE}")
    log_message(f"Model Params: CNN_Out={CNN_OUTPUT_DIM}, LSTM_Hidden={LSTM_HIDDEN_DIM}, LSTM_Layers={LSTM_LAYERS}, Dropout={DROPOUT}")
    log_message(f"Manifest: {MANIFEST_PATH}, Model Save Dir: {MODEL_SAVE_DIR}, Log File: {LOG_FILE}")

    # --- Data Loaders ---
    log_message("\n--- Loading Data ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(MANIFEST_PATH, char_to_int, BATCH_SIZE, test_split=0.1, val_split=0.1)
        log_message(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        log_message(f"FATAL Error loading data: {e}")
        exit()

    # --- Model Initialization ---
    log_message("\n--- Initializing Model ---")
    # Instantiate the model using parameters defined above
    model = LipReadModel(
        input_channels=1, # Grayscale
        cnn_output_dim=CNN_OUTPUT_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        num_classes=VOCAB_SIZE, # From dataset.py
        dropout_p=DROPOUT
    ).to(DEVICE)

    # Log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model initialized. Trainable Parameters: {num_params:,}")
    # Uncomment to print model structure (can be very long)
    # log_message("Model Architecture:")
    # log_message(str(model))

    # --- Loss Function ---
    # Use blank index from char_to_int mapping
    blank_idx = char_to_int.get(' ', 0) # Default to 0 if space not found, but it should be
    if blank_idx != 0:
        log_message(f"Warning: Blank character (' ') is not at index 0 in CHARSET. Check dataset.py. Using index {blank_idx}.")
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    log_message(f"CTC Loss initialized with blank index: {blank_idx}")

    # --- Optimizer ---
    # AdamW is often a good default choice
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    log_message(f"Optimizer: AdamW, Learning Rate: {LEARNING_RATE}")

    # --- Learning Rate Scheduler (Optional but recommended) ---
    # Reduces LR if validation CER plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2, verbose=True)
    log_message(f"Scheduler: ReduceLROnPlateau (monitors Val CER, factor=0.5, patience={PATIENCE // 2})")


    # --- Training Loop ---
    log_message("\n--- Starting Training Loop ---")
    best_val_cer = float('inf')
    epochs_no_improve = 0
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, DEVICE, epoch)

        # Evaluate on validation set
        val_loss, val_cer = evaluate(model, val_loader, ctc_loss, DEVICE, int_to_char, epoch)

        epoch_duration = time.time() - epoch_start_time

        # Log epoch results
        log_message(f"Epoch {epoch+1}/{EPOCHS} Summary | Duration: {epoch_duration:.2f}s")
        log_message(f"  Train Loss: {train_loss:.4f}")
        log_message(f"  Val Loss:   {val_loss:.4f}")
        log_message(f"  Val CER:    {val_cer:.4f}")

        # Step the learning rate scheduler based on validation CER
        scheduler.step(val_cer)

        # --- Check for Improvement and Save Best Model ---
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
            try:
                torch.save(model.state_dict(), model_path)
                log_message(f"----> Validation CER improved to {val_cer:.4f}! Model saved to {model_path}")
            except Exception as e:
                log_message(f"Error saving model: {e}")
            epochs_no_improve = 0 # Reset patience counter
        else:
            epochs_no_improve += 1
            log_message(f"Validation CER did not improve. ({epochs_no_improve}/{PATIENCE})")

        # --- Early Stopping Check ---
        if epochs_no_improve >= PATIENCE:
            log_message(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation CER.")
            break

    # --- End of Training ---
    total_training_time = time.time() - training_start_time
    log_message(f"\n--- Training Finished ---")
    log_message(f"Total Training Time: {total_training_time / 3600:.2f} hours ({total_training_time:.2f} seconds)")
    log_message(f"Best Validation CER achieved: {best_val_cer:.4f}")
    log_message(f"Model corresponding to best validation CER saved at: {os.path.join(MODEL_SAVE_DIR, MODEL_NAME)}")

    # --- Final Evaluation on Test Set ---
    log_message("\n--- Evaluating on Test Set using Best Model ---")
    # Load the best saved model weights
    try:
        best_model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
        if os.path.exists(best_model_path):
             model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
             log_message(f"Loaded best model weights from {best_model_path} for final test evaluation.")
             test_loss, test_cer = evaluate(model, test_loader, ctc_loss, DEVICE, int_to_char, -1) # Use -1 epoch num for test
             log_message("-" * 30)
             log_message(f"Final Test Set Results:")
             log_message(f"  Test Loss: {test_loss:.4f}")
             log_message(f"  Test CER:  {test_cer:.4f}")
             log_message("-" * 30)
        else:
             log_message(f"Warning: Best model file not found at {best_model_path}. Cannot evaluate on test set.")
    except Exception as e:
        log_message(f"Error loading best model or evaluating on test set: {e}")

    log_message("--- Training Script Completed ---")