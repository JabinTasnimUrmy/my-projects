# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader # Corrected DataLoader import
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import os

# --- Character Set Definition ---
# Define the vocabulary based on GRID dataset characters.
# IMPORTANT: Ensure the 'blank' token (space) is at index 0.
CHARSET = " abcdefghijklmnopqrstuvwxyz0123456789" # Leading space is blank
char_to_int = {char: i for i, char in enumerate(CHARSET)}
int_to_char = {i: char for i, char in enumerate(CHARSET)}
VOCAB_SIZE = len(CHARSET) # Includes the blank token at index 0
BLANK_INDEX = char_to_int.get(' ', 0) # Get blank index (should be 0)

print(f"Vocabulary Size (including blank): {VOCAB_SIZE}")
print(f"Blank Index: {BLANK_INDEX}")

# --- PyTorch Dataset Class for GRID ---
class GridDataset(Dataset):
    def __init__(self, manifest_path, char_to_int_map):
        """
        Args:
            manifest_path (string): Path to the csv file with npy_path and transcript.
            char_to_int_map (dict): Dictionary mapping characters to integers.
        """
        try:
            self.manifest = pd.read_csv(manifest_path)
            self.char_to_int = char_to_int_map
            print(f"Loaded manifest '{manifest_path}': {len(self.manifest)} samples")
        except FileNotFoundError:
             print(f"FATAL Error: Manifest file not found at {manifest_path}")
             raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        except Exception as e:
             print(f"FATAL Error loading manifest {manifest_path}: {e}")
             raise e

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.manifest)

    def __getitem__(self, idx):
        """
        Loads and returns one sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (video_tensor, label_tensor, input_length, label_length)
                   Returns (None, None, None, None) if loading fails.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_row = self.manifest.iloc[idx]
        npy_path = data_row['npy_path']
        transcript = str(data_row.get('transcript', ''))

        # --- Load Video Frames ---
        try:
            video_frames = np.load(npy_path)
            expected_shape_prefix = (75, 1) # T=75, C=1
            if video_frames.ndim != 4 or video_frames.shape[0] != expected_shape_prefix[0] or video_frames.shape[1] != expected_shape_prefix[1]:
                # print(f"Warning: Unexpected video shape in {npy_path}. Skipping.")
                return None, None, None, None
            video_tensor = torch.FloatTensor(video_frames) # Shape: (T, C, H, W)
        except FileNotFoundError:
             # print(f"Error: File not found {npy_path}. Skipping sample.")
             return None, None, None, None
        except Exception as e:
             # print(f"Error loading or processing {npy_path}: {e}. Skipping sample.")
             return None, None, None, None

        # --- Encode Transcript ---
        label_indices = [self.char_to_int.get(char, -1) for char in transcript.lower()]
        label_indices = [l for l in label_indices if l != -1] # Filter unknown chars
        if not label_indices:
            # print(f"Warning: Transcript for {npy_path} became empty. Skipping.")
            return None, None, None, None
        label_tensor = torch.LongTensor(label_indices) # Shape: (L,)

        # --- Get Lengths ---
        input_length = torch.LongTensor([video_tensor.shape[0]]) # Shape: (1,) holding T
        label_length = torch.LongTensor([len(label_indices)]) # Shape: (1,) holding L

        return video_tensor, label_tensor, input_length, label_length

# --- Custom Collate Function for Padding Batches ---
def collate_fn(batch):
    """Pads sequences in a batch to the maximum length in that batch."""
    # Filter out None items resulting from loading errors in __getitem__
    batch = [item for item in batch if item[0] is not None]
    if not batch: # If the batch is empty after filtering
        return None, None, None, None

    # Unzip the batch into separate lists
    videos, labels, input_lengths, label_lengths = zip(*batch)

    # --- Pad Video Sequences ---
    try:
        C, H, W = videos[0].shape[1], videos[0].shape[2], videos[0].shape[3]
        videos_reshaped = [v.contiguous().view(v.size(0), -1) for v in videos] # List of (T, C*H*W)
        padded_videos_flat = pad_sequence(videos_reshaped, batch_first=True, padding_value=0.0) # (B, T_max, C*H*W)
        B, T_max, _ = padded_videos_flat.shape
        videos_batch = padded_videos_flat.view(B, T_max, C, H, W) # (B, T_max, C, H, W)
    except Exception as e:
        print(f"Error during video padding: {e}")
        print(f"Video shapes in batch: {[v.shape for v in videos]}")
        return None, None, None, None # Cannot proceed if padding fails

    # --- Pad Label Sequences ---
    labels_batch = pad_sequence(labels, batch_first=True, padding_value=BLANK_INDEX) # (B, L_max)

    # --- Concatenate Length Tensors ---
    input_lengths_batch = torch.cat(input_lengths) # (B,)
    label_lengths_batch = torch.cat(label_lengths) # (B,)

    return videos_batch, labels_batch, input_lengths_batch, label_lengths_batch

# --- Function to Create DataLoaders with Splitting ---
def get_dataloaders(manifest_path, char_to_int_map, batch_size, test_split=0.1, val_split=0.1, num_workers=4, pin_memory=True):
    """Creates train, validation, and test DataLoaders from a single manifest file."""
    from sklearn.model_selection import train_test_split

    print(f"Creating dataloaders from: {manifest_path}")
    try:
        df = pd.read_csv(manifest_path)
        if df.empty: raise ValueError("Manifest file is empty.")
    except Exception as e:
        print(f"Failed to load or validate manifest file {manifest_path}: {e}")
        raise

    # Validate split ratios
    if not (0 < test_split < 1) or not (0 < val_split < 1) or test_split + val_split >= 1.0:
        raise ValueError("Invalid split ratios. Ensure test_split and val_split are > 0 and their sum < 1.")

    # Perform splits
    train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=42, shuffle=True)
    val_size_adjusted = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=42, shuffle=True)

    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more data splits (train, val, test) are empty. Check manifest size and split ratios.")

    # --- Inner Dataset class to accept DataFrame ---
    class GridDatasetFromDF(GridDataset):
         def __init__(self, dataframe, char_to_int_map):
             # Override __init__ to use DataFrame instead of path
             self.manifest = dataframe
             self.char_to_int = char_to_int_map
             # Suppress print for splits if desired:
             # print(f"Initialized dataset from DataFrame: {len(self.manifest)} samples")
    # --- Create Datasets from DataFrames ---
    try:
        train_dataset = GridDatasetFromDF(train_df, char_to_int_map)
        val_dataset = GridDatasetFromDF(val_df, char_to_int_map)
        test_dataset = GridDatasetFromDF(test_df, char_to_int_map)
    except Exception as e:
        print(f"Error creating dataset instances from DataFrames: {e}")
        raise

    # --- Create DataLoaders ---
    effective_pin_memory = pin_memory and (torch.cuda.is_available())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers,
                              pin_memory=effective_pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=effective_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=effective_pin_memory)

    return train_loader, val_loader, test_loader

# --- Example Usage (for testing dataset.py itself) ---
if __name__ == '__main__':
    print("Testing dataset loading and batching...")
    manifest = 'grid_manifest.csv'
    if not os.path.exists(manifest):
        print(f"Error: Cannot test dataset.py - {manifest} not found.")
    else:
        try:
            # Test with num_workers=0 for easier debugging if needed
            train_dl, val_dl, test_dl = get_dataloaders(manifest, char_to_int, batch_size=4, test_split=0.1, val_split=0.1, num_workers=0)

            print("\nTesting Train Loader (first 2 batches)...")
            for i, batch in enumerate(train_dl):
                if batch[0] is None:
                     print(f"Skipped potentially empty batch {i}")
                     continue
                videos, labels, input_lens, label_lens = batch
                print(f" Batch {i+1}:")
                print(f"  Videos shape: {videos.shape}") # Should be (B, T=75, C=1, H=96, W=96)
                print(f"  Labels shape: {labels.shape}") # Should be (B, L_max)
                print(f"  Input lengths: {input_lens.tolist()}") # List of T=75
                print(f"  Label lengths: {label_lens.tolist()}") # List of actual label lengths
                # ---- THE PROBLEMATIC LINE IS REMOVED HERE ----
                if i >= 1: # Only check first 2 batches
                    break
            print("\nDataset and DataLoader test logic completed.")

        except Exception as e:
            print(f"\nError during dataset testing: {e}")
            import traceback
            traceback.print_exc()