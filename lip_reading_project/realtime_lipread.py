# realtime_lipread.py

import cv2
import torch
import numpy as np
import dlib
import time
import torch.nn.functional as F
import os # <--- ADD THIS LINE TO IMPORT THE OS MODULE

# Import custom modules - ensure these files are in the same directory
try:
    from model import LipReadModel
    from dataset import int_to_char, char_to_int, VOCAB_SIZE # Needs VOCAB_SIZE for model init
    from train import greedy_decoder # Import the decoder used during training/evaluation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure model.py, dataset.py, and train.py are in the same directory.")
    exit()

# --- Configuration ---
# Path to your trained model weights
MODEL_PATH = 'saved_models/lipread_cnn_lstm_ctc_grid.pth'
# Path to dlib's facial landmark predictor
DLIB_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
# Set device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing settings (MUST MATCH TRAINING PREPROCESSING)
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 1 # Grayscale
MOUTH_LANDMARK_INDICES = list(range(48, 68)) # Indices for mouth landmarks in dlib's 68 points
MOUTH_PADDING_RATIO = 0.3 # Padding around mouth box (should match preprocessing)

# Real-time settings
SEQUENCE_LENGTH = 75 # Number of frames the model was trained on (GRID specific)
WEBCAM_ID = 0        # Default webcam ID (change if you have multiple)
DISPLAY_SCALE = 1.0  # Scale factor for the display window size (adjust if needed)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0) # Green
LINE_TYPE = 2
PREDICTION_DISPLAY_POS = (10, 30) # Top-left corner for text display
ROI_RECT_COLOR = (0, 0, 255) # Red rectangle for mouth ROI
LANDMARK_COLOR = (0, 255, 0) # Green dots for landmarks (optional)
DRAW_LANDMARKS = False # Set to True to draw all 68 landmarks

# Model parameters (MUST MATCH THE SAVED MODEL'S ARCHITECTURE)
CNN_OUTPUT_DIM = 512
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
DROPOUT = 0.3 # Note: Dropout is automatically disabled in model.eval() mode

# --- Model Loading ---
print("Loading model...")
# Instantiate the model with the same parameters used during training
model = LipReadModel(
    input_channels=IMG_CHANNELS,
    cnn_output_dim=CNN_OUTPUT_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    lstm_layers=LSTM_LAYERS,
    num_classes=VOCAB_SIZE, # VOCAB_SIZE imported from dataset.py
    dropout_p=DROPOUT
).to(DEVICE)

# Load the saved weights
try:
    # Use os.path.exists now that 'os' is imported
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model weights from {MODEL_PATH}: {e}")
    print("Ensure the model architecture defined in model.py matches the saved weights.")
    exit()

# --- Load dlib ---
print("Loading dlib detector and predictor...")
try:
    # Use os.path.exists now that 'os' is imported
    if not os.path.exists(DLIB_PREDICTOR_PATH):
        raise FileNotFoundError(f"Dlib predictor file not found at {DLIB_PREDICTOR_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    print("Dlib models loaded.")
except Exception as e:
    print(f"Error loading dlib models: {e}")
    print(f"Ensure '{DLIB_PREDICTOR_PATH}' exists in the project directory.")
    exit()

# --- Helper: Preprocess Frame (Similar to training preprocessing) ---
def preprocess_frame(frame_bgr):
    """
    Detects face, extracts landmarks, isolates, preprocesses mouth ROI.
    Args:
        frame_bgr (np.ndarray): Input frame in BGR format from webcam.
    Returns:
        tuple: (processed_roi, roi_coords)
               processed_roi (np.ndarray): Normalized grayscale mouth ROI (C, H, W) or None.
               roi_coords (tuple): Coordinates (x1, y1, x2, y2) for drawing box or None.
    """
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray, 0) # Use 0 for faster detection (no upsampling)

    if faces:
        try:
            shape = predictor(frame_gray, faces[0]) # Assume first face is target
            coords = np.zeros((68, 2), dtype=int)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            if DRAW_LANDMARKS: # Draw landmarks if enabled
                for i in range(68):
                    cv2.circle(frame_bgr, (coords[i][0], coords[i][1]), 1, LANDMARK_COLOR, -1)

            # Extract Mouth ROI using landmark indices
            mouth_coords = coords[MOUTH_LANDMARK_INDICES]
            if not mouth_coords.size: return None, None

            # Basic check for valid coordinates
            if np.any(mouth_coords < 0) or np.any(mouth_coords[:, 0] >= frame_gray.shape[1]) or np.any(mouth_coords[:, 1] >= frame_gray.shape[0]):
                 return None, None

            x_min, y_min = np.min(mouth_coords, axis=0)
            x_max, y_max = np.max(mouth_coords, axis=0)

            # Add padding (same logic as preprocessing script)
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0: return None, None

            pad_w = int(width * MOUTH_PADDING_RATIO / 2)
            pad_h = int(height * MOUTH_PADDING_RATIO / 2)

            x1 = max(0, x_min - pad_w)
            y1 = max(0, y_min - pad_h)
            x2 = min(frame_gray.shape[1], x_max + pad_w)
            y2 = min(frame_gray.shape[0], y_max + pad_h)

            if y1 >= y2 or x1 >= x2: return None, None

            # Crop, Resize, Normalize
            mouth_roi = frame_gray[y1:y2, x1:x2]
            if mouth_roi.size == 0: return None, None

            resized_roi = cv2.resize(mouth_roi, (IMG_WIDTH, IMG_HEIGHT))
            normalized_roi = resized_roi / 255.0

            # Add channel dimension: (H, W) -> (C, H, W) and set dtype
            processed_roi = np.expand_dims(normalized_roi, axis=0).astype(np.float32) # Shape (1, H, W)

            roi_coords = (x1, y1, x2, y2) # Coordinates for drawing rectangle
            return processed_roi, roi_coords

        except Exception as e:
            # print(f"Error during landmark prediction or ROI extraction: {e}")
            return None, None # Return None if any error occurs in detection/processing
    else:
        return None, None # No face detected

# --- Webcam Setup ---
print("Opening webcam...")
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Could not open webcam ID {WEBCAM_ID}.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Optional: Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Optional: Set frame height
print(f"Webcam opened (ID: {WEBCAM_ID}). Press 'q' to quit.")

# --- Real-Time Loop Variables ---
frame_buffer = [] # Stores the sequence of processed mouth ROIs
current_prediction = ""
last_valid_roi_np = np.zeros((IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32) # Placeholder if detection fails early

# --- Main Inference Loop ---
while True:
    ret, frame_bgr = cap.read() # Read frame in BGR
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        time.sleep(0.5) # Wait a bit before trying again
        continue

    display_frame = frame_bgr.copy() # Create a copy for displaying results

    # Preprocess the current frame
    processed_roi_np, roi_coords = preprocess_frame(display_frame) # Pass display frame if drawing landmarks

    if processed_roi_np is not None:
        # Append the valid processed frame to the buffer
        frame_buffer.append(processed_roi_np)
        last_valid_roi_np = processed_roi_np # Update the last known good ROI
        if roi_coords: # Draw rectangle if coordinates are available
             cv2.rectangle(display_frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), ROI_RECT_COLOR, 2)
    elif len(frame_buffer) > 0:
        # If face/mouth detection fails, reuse the *last valid* ROI to keep sequence length
        frame_buffer.append(last_valid_roi_np)
    else:
        # If buffer is empty AND detection fails, append zeros
         frame_buffer.append(np.zeros((IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32))

    # Maintain the buffer size to match the required sequence length
    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0) # Remove the oldest frame

    # --- Prediction Step (only when buffer is full) ---
    if len(frame_buffer) == SEQUENCE_LENGTH:
        # Prepare input tensor for the model
        # 1. Stack the list of (C, H, W) numpy arrays -> (T, C, H, W) numpy array
        input_sequence_np = np.stack(frame_buffer, axis=0)
        # 2. Convert to PyTorch tensor -> (T, C, H, W)
        input_tensor = torch.from_numpy(input_sequence_np)
        # 3. Add batch dimension -> (B=1, T, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)
        # 4. Move tensor to the target device (GPU or CPU)
        input_tensor = input_tensor.to(DEVICE)

        # Perform inference
        with torch.no_grad(): # Disable gradient calculation
            log_probs = model(input_tensor) # Output shape: (T, B=1, N)

        # Decode the output using the same greedy decoder as in training/evaluation
        # Note: log_probs is (T, B=1, N), decoder expects (T, B, N)
        decoded_texts = greedy_decoder(log_probs, int_to_char)
        current_prediction = decoded_texts[0] if decoded_texts else "" # Get prediction for the single batch item

    # --- Display Results ---
    # Resize frame for display if needed
    if DISPLAY_SCALE != 1.0:
        scaled_h = int(display_frame.shape[0] * DISPLAY_SCALE)
        scaled_w = int(display_frame.shape[1] * DISPLAY_SCALE)
        display_frame_scaled = cv2.resize(display_frame, (scaled_w, scaled_h))
    else:
        display_frame_scaled = display_frame

    # Add prediction text onto the frame
    cv2.putText(display_frame_scaled, f"Prediction: {current_prediction}",
                PREDICTION_DISPLAY_POS, FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    # Show the frame
    cv2.imshow('Real-Time Lip Reading - Press Q to Quit', display_frame_scaled)

    # --- Exit Condition ---
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

# --- Cleanup ---
print("Closing webcam and destroying windows...")
cap.release() # Release the webcam resource
cv2.destroyAllWindows() # Close all OpenCV windows
print("Script finished.")