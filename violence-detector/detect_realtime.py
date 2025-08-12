import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import collections
import numpy as np

# Import the custom model class
from models.violence_model import ViolenceClassifier

# Configuration 
MODEL_PATH = "checkpoints/violence_model_best.pth"
SEQUENCE_LENGTH = 16
IMAGE_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# Model Loading 
model = ViolenceClassifier(dropout_p=0.5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)
print("[INFO] Model loaded successfully.")

# Preprocessing 
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Real-Time Detection Logic 
frames_queue = collections.deque(maxlen=SEQUENCE_LENGTH)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("[ERROR] Cannot open webcam. Please check its connection.")
    exit()

print("[INFO] Webcam opened. Starting real-time detection...")
print("[INFO] Press 'q' to quit.")

prediction_text = "Waiting for buffer..."
prediction_prob = 0.0

while True:
    ret, frame = webcam.read()
    if not ret:
        print("[INFO] End of stream or webcam disconnected.")
        break

    # Frame Preprocessing 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    preprocessed_frame = data_transform(pil_image)
    frames_queue.append(preprocessed_frame)

    # Prediction 
    if len(frames_queue) == SEQUENCE_LENGTH:
        # Prepare the input tensor for the model
        # The shape from stacking is (T, C, H, W).
        # .unsqueeze(0) adds the batch dimension to make it (B, T, C, H, W).
        input_tensor = torch.stack(list(frames_queue), dim=0).unsqueeze(0).to(DEVICE)
        
        
        # Make prediction with no gradient tracking
        with torch.no_grad():
            logit = model(input_tensor)
            probability = torch.sigmoid(logit).item()

        prediction_prob = probability
        if prediction_prob > 0.70:  # Threshold for displaying "Fight"
            prediction_text = "Fight"
        else:
            prediction_text = "NonFight"

    # Display Results on Frame 
    display_color = (0, 0, 255) if prediction_text == "Fight" else (0, 255, 0)
    display_text = f"{prediction_text}: {prediction_prob:.2f}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
    
    cv2.imshow("Real-time Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup 
webcam.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped and resources released.")