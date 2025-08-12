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
# Paths to the model and video files
MODEL_PATH = "checkpoints/violence_model_best.pth"
INPUT_VIDEO_PATH = "test_video.mp4" 
OUTPUT_VIDEO_PATH = "output_demo.mp4"

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

# Video Processing Logic 
frames_queue = collections.deque(maxlen=SEQUENCE_LENGTH)
video_capture = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot open video file: {INPUT_VIDEO_PATH}")
    exit()

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"[INFO] Processing video file: {INPUT_VIDEO_PATH}")

prediction_text = "NonFight"
prediction_prob = 0.0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[INFO] End of video file reached.")
        break

    rgb_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocessed_frame = data_transform(rgb_frame_pil)
    frames_queue.append(preprocessed_frame)

    if len(frames_queue) == SEQUENCE_LENGTH:
        input_tensor = torch.stack(list(frames_queue), dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(input_tensor)
            probability = torch.sigmoid(logit).item()
        prediction_prob = probability
        prediction_text = "Fight" if probability > 0.70 else "NonFight"

    
    # Display Results and Write to Output Video 
    display_color = (0, 0, 255) if prediction_text == "Fight" else (0, 255, 0)
    display_text = f"{prediction_text}: {prediction_prob:.2f}"

    # font properties
    font_scale = 3.5  
    font_thickness = 5 
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (50, 120) 

    # text box size
    (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
    
    
    rect_start = (text_position[0] - 20, text_position[1] - text_height - 20)
    rect_end = (text_position[0] + text_width + 20, text_position[1] + baseline)
    overlay = frame.copy()
    cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw the text on the frame
    cv2.putText(frame, display_text, text_position, font, font_scale, display_color, font_thickness, cv2.LINE_AA)
    


    video_writer.write(frame)
    cv2.imshow("Video Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"[SUCCESS] Finished processing. Output video saved to: {OUTPUT_VIDEO_PATH}")