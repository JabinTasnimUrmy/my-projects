import torch
from torchsummary import summary
from models.violence_model import ViolenceClassifier

# --- Configuration ---
DEVICE = torch.device("cpu")

# --- Model Loading ---
# Instantiate our full model and move it to the CPU for inspection
model = ViolenceClassifier(dropout_p=0.5)
model = model.to(DEVICE)

print("="*80)
print("       Model Architecture Summary: R(2+1)D-18 (Fine-Tuned)")
print("="*80)
print("This summary shows the layers of the core R(2+1)D-18 model, including our")
print("custom fine-tuned classifier attached as the final 'fc' layer.\n")
print("The model expects an input of shape (Batch, Channels, Time, Height, Width).")
print("For this summary, we will use a sample input shape of (1, 3, 16, 112, 112).\n")

# --- Generate and Print the Summary ---
# The input shape (Channels, Time, Height, Width) for the summary tool
input_shape = (3, 16, 112, 112)

# FIX: Explicitly tell torchsummary to run on the CPU to match the model's device.
summary(model.model, input_shape, device=DEVICE.type)

print("\n" + "="*80)
print("The summary above reflects the complete, fine-tuned model.")
print("The final 'fc' layer containing a Sequential block is our custom classifier.")
print("="*80)