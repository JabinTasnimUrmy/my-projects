# visualize_model.py

import torch
from torchsummary import summary 
import sys

# Import the model definition from model.py file
try:
    from model import LipReadModel
    from dataset import VOCAB_SIZE
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure model.py and dataset.py are in the same directory.")
    exit()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters 
CNN_OUTPUT_DIM = 512
LSTM_HIDDEN_DIM = 256
LSTM_LAYERS = 2
DROPOUT = 0.3



INPUT_SHAPE_2D = (INPUT_CHANNELS, 96, 96) 
SEQUENCE_LENGTH = 75 


# Instantiate the Model 
print("--- Initializing Model for Visualization ---")
model = LipReadModel(
    input_channels=INPUT_CHANNELS,
    cnn_output_dim=CNN_OUTPUT_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    lstm_layers=LSTM_LAYERS,
    num_classes=VOCAB_SIZE,
    dropout_p=DROPOUT
).to(DEVICE)

# Generate and Print the Full Model Structure 
print("\n" + "="*80)
print(" Full PyTorch Model Structure (model.__str__) ".center(80, "="))
print("="*80)
print(model)
print("="*80)


# Generate Layer-by-Layer Summary using torchsummary 

print("\n" + "="*80)
print(" CNN Backbone Summary (torchsummary) ".center(80, "="))
print("="*80)
print(f"Input shape to CNN (per frame): {INPUT_SHAPE_2D}")

# Extract the CNN backbone from the model
cnn_backbone = torch.nn.Sequential(
    model.conv1,
    model.bn1,
    torch.nn.ReLU(),
    model.pool1,
    model.conv2,
    model.bn2,
    torch.nn.ReLU(),
    model.pool2,
    model.conv3,
    model.bn3,
    torch.nn.ReLU(),
    model.pool3,
    model.conv4,
    model.bn4,
    torch.nn.ReLU(),
    model.pool4,
    torch.nn.Flatten(),
    model.cnn_fc,
    torch.nn.ReLU(),
    model.cnn_dropout
).to(DEVICE)

# Use summary on the CNN part

summary(cnn_backbone, input_size=INPUT_SHAPE_2D, device=DEVICE.type)

# Manually Describe the Rest of the Architecture 
print("\n" + "="*80)
print(" Full Model Data Flow Description ".center(80, "="))
print("="*80)

description = f"""
1. Input Video Sequence:
   - Shape: (Batch, Time={SEQUENCE_LENGTH}, Channels={INPUT_CHANNELS}, Height=96, Width=96)
   - The input is a batch of video clips, each with {SEQUENCE_LENGTH} grayscale frames.

2. CNN Backbone (Spatial Feature Extraction):
   - The video sequence is reshaped to (Batch * Time, Channels, Height, Width) to process each frame.
   - Each frame is passed through the CNN backbone summarized above.
   - Output of CNN per frame: A feature vector of size {CNN_OUTPUT_DIM}.

3. Sequence Reshaping for LSTM:
   - The frame-wise feature vectors are re-assembled into sequences.
   - Shape changes from (Batch * Time, {CNN_OUTPUT_DIM}) back to (Batch, Time={SEQUENCE_LENGTH}, {CNN_OUTPUT_DIM}).

4. Bidirectional LSTM (Temporal Modeling):
   - Type: {type(model.lstm)}
   - Takes the sequence of {SEQUENCE_LENGTH} feature vectors as input.
   - Hidden Size: {LSTM_HIDDEN_DIM} (per direction)
   - Num Layers: {LSTM_LAYERS}
   - Bidirectional: True (processes sequence forwards and backwards)
   - Output Shape: (Batch, Time={SEQUENCE_LENGTH}, Features={LSTM_HIDDEN_DIM * 2})

5. Classifier (Frame-wise Prediction):
   - Type: {type(model.fc_classifier)}
   - A fully connected layer that maps the LSTM output to class logits for each time step.
   - Input Features: {LSTM_HIDDEN_DIM * 2}
   - Output Features (Num Classes): {VOCAB_SIZE}
   - Output Shape: (Batch, Time={SEQUENCE_LENGTH}, Classes={VOCAB_SIZE})

6. Final Output Transformation for CTC:
   - A LogSoftmax activation is applied to the logits.
   - The tensor dimensions are permuted from (Batch, Time, Classes) to (Time, Batch, Classes).
   - Final Output Shape for CTC Loss/Decoding: ({SEQUENCE_LENGTH}, Batch, {VOCAB_SIZE})
"""
print(description)
print("="*80)


# Save the model architecture details to a text file
output_filename = 'model_architecture.txt'
print(f"\nSaving this summary to '{output_filename}'...")
with open(output_filename, 'w') as f:
   
    from io import StringIO
    
  
    f.write("="*80 + "\n")
    f.write(" Full PyTorch Model Structure (model.__str__) ".center(80, "=") + "\n")
    f.write("="*80 + "\n")
    f.write(str(model) + "\n")
    f.write("="*80 + "\n\n")
    
    
    f.write("="*80 + "\n")
    f.write(" CNN Backbone Summary (torchsummary) ".center(80, "=") + "\n")
    f.write("="*80 + "\n")
    
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    summary(cnn_backbone, input_size=INPUT_SHAPE_2D, device=DEVICE.type)
    sys.stdout = old_stdout 
    
    f.write(captured_output.getvalue())
    f.write("\n")


    f.write("="*80 + "\n")
    f.write(" Full Model Data Flow Description ".center(80, "=") + "\n")
    f.write("="*80 + "\n")
    f.write(description)
    f.write("="*80 + "\n")

print(f"Successfully saved model architecture details to '{output_filename}'.")