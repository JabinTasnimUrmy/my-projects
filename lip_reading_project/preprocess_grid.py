import os
import cv2
import dlib
import numpy as np
import re
import pandas as pd
from tqdm import tqdm # Progress bar

# --- Configuration ---
GRID_CORPUS_PATH = 'data' # Root folder containing sX_processed folders
OUTPUT_DIR = 'processed_grid_data' # Where to save processed numpy arrays and manifest
DLIB_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' # Should be in the project root
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 1 # Grayscale
SEQUENCE_LENGTH = 75 # GRID videos are fixed length (3 seconds * 25 fps = 75 frames)
MOUTH_LANDMARK_INDICES = list(range(48, 68))
MOUTH_PADDING_RATIO = 0.3 # Increase ROI size by 30% around mouth landmarks

# --- Initialize dlib ---
print("Initializing dlib...")
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
except Exception as e:
    print(f"FATAL Error loading dlib predictor from {DLIB_PREDICTOR_PATH}: {e}")
    print(f"Ensure '{DLIB_PREDICTOR_PATH}' is in the project directory.")
    exit()
print("dlib initialized.")

# --- Helper Function: Extract Mouth ROI ---
def get_mouth_roi(frame_gray, shape):
    # Extracts mouth landmarks
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    mouth_coords = coords[MOUTH_LANDMARK_INDICES]
    if not mouth_coords.size: return None

    # Check for invalid coordinates
    if np.any(mouth_coords < 0) or np.any(mouth_coords[:, 0] >= frame_gray.shape[1]) or np.any(mouth_coords[:, 1] >= frame_gray.shape[0]):
         return None

    # Calculate bounding box
    x_min, y_min = np.min(mouth_coords, axis=0)
    x_max, y_max = np.max(mouth_coords, axis=0)

    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0: return None

    pad_w = int(width * MOUTH_PADDING_RATIO / 2)
    pad_h = int(height * MOUTH_PADDING_RATIO / 2)

    x1 = max(0, x_min - pad_w)
    y1 = max(0, y_min - pad_h)
    x2 = min(frame_gray.shape[1], x_max + pad_w)
    y2 = min(frame_gray.shape[0], y_max + pad_h)

    if y1 >= y2 or x1 >= x2: return None

    # Crop, Resize, Normalize
    mouth_roi = frame_gray[y1:y2, x1:x2]
    if mouth_roi.size == 0: return None

    try:
        resized_roi = cv2.resize(mouth_roi, (IMG_WIDTH, IMG_HEIGHT))
        normalized_roi = resized_roi / 255.0
        return normalized_roi
    except cv2.error:
        return None


# --- Main Preprocessing Loop ---
processed_samples = []
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if GRID_CORPUS_PATH exists
if not os.path.isdir(GRID_CORPUS_PATH):
    print(f"FATAL Error: Dataset directory '{GRID_CORPUS_PATH}' not found.")
    exit()

# Find speaker folders (e.g., s1_processed)
try:
    # Find directories inside 'data' that likely represent speakers
    speaker_folders = sorted([
        d for d in os.listdir(GRID_CORPUS_PATH)
        if os.path.isdir(os.path.join(GRID_CORPUS_PATH, d)) and d.startswith('s') and d.endswith('_processed')
    ])
except FileNotFoundError:
    print(f"FATAL Error: Cannot list directory '{GRID_CORPUS_PATH}'. Check path and permissions.")
    exit()

if not speaker_folders:
    print(f"FATAL Error: No speaker folders matching 's*_processed' found inside '{GRID_CORPUS_PATH}'.")
    print("Check the dataset structure. Expecting folders like 's1_processed', 's2_processed', etc.")
    exit()

print(f"Found {len(speaker_folders)} speaker folders in '{GRID_CORPUS_PATH}'. Starting processing...")

total_videos_processed = 0
total_videos_skipped_frames = 0
total_videos_skipped_align = 0
total_videos_skipped_no_words = 0
total_videos_open_error = 0
total_numpy_save_errors = 0

for speaker_folder_name in tqdm(speaker_folders, desc="Processing Speakers"):
    speaker_path = os.path.join(GRID_CORPUS_PATH, speaker_folder_name)
    align_path = os.path.join(speaker_path, 'align')

    # Check if the 'align' subfolder exists for this speaker
    if not os.path.exists(align_path):
        # print(f"Warning: Skipping {speaker_folder_name}: 'align' subfolder missing at {align_path}.")
        continue

    # Find video files directly within the speaker folder
    try:
        all_files_in_speaker_path = os.listdir(speaker_path)
        video_files = sorted([f for f in all_files_in_speaker_path if f.endswith('.mpg')])
        if not video_files:
            # print(f"Warning: No .mpg files found directly in {speaker_path}, skipping speaker.")
            continue
    except FileNotFoundError:
        # print(f"Warning: Error listing directory {speaker_path}, skipping.")
        continue

    # Extract speaker ID (e.g., 's1') from folder name ('s1_processed')
    speaker_id_match = re.match(r'(s\d+)', speaker_folder_name)
    speaker = speaker_id_match.group(1) if speaker_id_match else speaker_folder_name # Fallback

    for video_file in video_files:
        base_name = os.path.splitext(video_file)[0]
        video_file_path = os.path.join(speaker_path, video_file) # Video path is direct
        align_file_path = os.path.join(align_path, base_name + '.align') # Align path uses subfolder

        # 1. Check and Read Alignment File
        if not os.path.exists(align_file_path):
            total_videos_skipped_align += 1
            continue

        transcript = ""
        try:
            with open(align_file_path, 'r') as f:
                words = [parts[2] for line in f for parts in [line.strip().split()] if len(parts) == 3 and parts[2] not in ['sil', 'sp']]
            transcript = " ".join(words)
        except Exception as e:
            print(f"Warning: Error reading align file {align_file_path}: {e}. Skipping video.")
            continue

        if not transcript:
            total_videos_skipped_no_words += 1
            continue

        # 2. Process Video Frames
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            # print(f"Warning: Could not open video file {video_file_path}, skipping.")
            total_videos_open_error += 1
            continue

        frames_data = []
        frame_count = 0
        last_roi = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32) # Initialize last_roi with zeros

        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break # Video ended

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(frame_gray, 0)

            current_roi = None
            if faces:
                try:
                    shape = predictor(frame_gray, faces[0])
                    current_roi = get_mouth_roi(frame_gray, shape)
                except Exception: # Catch potential dlib errors
                    current_roi = None

            if current_roi is not None:
                frames_data.append(current_roi)
                last_roi = current_roi # Update last known good ROI
            else:
                frames_data.append(last_roi) # Reuse last ROI (or initial zeros) if detection fails

            frame_count += 1
        cap.release()

        # Pad if video was shorter than SEQUENCE_LENGTH
        while len(frames_data) < SEQUENCE_LENGTH:
            frames_data.append(last_roi) # Pad with the last valid ROI (or zeros)

        # 3. Verify and Save Processed Data
        if len(frames_data) == SEQUENCE_LENGTH:
            try:
                video_array = np.array(frames_data, dtype=np.float32)
                # Add channel dimension: (T, H, W) -> (T, C, H, W)
                video_array = np.expand_dims(video_array, axis=1)

                # Final shape check
                if video_array.shape != (SEQUENCE_LENGTH, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
                     # print(f"Warning: Incorrect final shape for {base_name}. Expected {(SEQUENCE_LENGTH, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)}, got {video_array.shape}. Skipping.")
                     total_videos_skipped_frames += 1
                     continue

                # Save .npy file (use speaker ID in filename)
                output_npy_filename = f"{speaker}_{base_name}.npy"
                output_npy_path = os.path.join(OUTPUT_DIR, output_npy_filename)
                np.save(output_npy_path, video_array)

                # Store info for manifest
                processed_samples.append({
                    'npy_path': output_npy_path, # Relative path to output dir
                    'speaker': speaker,
                    'video_id': base_name,
                    'transcript': transcript.lower() # Normalize case
                })
                total_videos_processed += 1
            except Exception as e:
                 print(f"Warning: Error stacking or saving numpy array for {video_file_path}: {e}. Skipping.")
                 total_numpy_save_errors += 1
        else:
             # This signifies an issue with the frame counting/padding logic
             # print(f"Critical Warning: Frame count mismatch for {video_file} ({len(frames_data)} != {SEQUENCE_LENGTH}). Skipping.")
             total_videos_skipped_frames += 1

# 4. Create Manifest File
manifest_df = pd.DataFrame(processed_samples)
manifest_path = 'grid_manifest.csv'
try:
    if not manifest_df.empty:
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\nManifest file with {len(manifest_df)} entries saved to: {manifest_path}")
    else:
        print("\nNo samples were successfully processed. Manifest file not created.")
except Exception as e:
    print(f"Error saving manifest file {manifest_path}: {e}")


print(f"\n--- Preprocessing Summary ---")
print(f"Speaker folders found: {len(speaker_folders)}")
print(f"Successfully processed and saved: {total_videos_processed} videos.")
print(f"Skipped (Missing .align file): {total_videos_skipped_align}")
print(f"Skipped (No words in .align): {total_videos_skipped_no_words}")
print(f"Skipped (Video open error): {total_videos_open_error}")
print(f"Skipped (Frame count/shape issue): {total_videos_skipped_frames}")
print(f"Skipped (NumPy save error): {total_numpy_save_errors}")
print("-" * 30)