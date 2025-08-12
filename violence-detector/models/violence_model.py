

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

def get_video_model(dropout_p=0.5):
    """
    Loads the R(2+1)D-18 model, pre-trained on the Kinetics-400 dataset,
    and replaces the final classification layer for our binary task.
    """
    # Load the pre-trained R(2+1)D-18 model
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3), 
        nn.Linear(256, 1)
    )
    return model

class ViolenceClassifier(nn.Module):
    """
    Wrapper for the R(2+1)D model to ensure the input tensor shape is correct.
    """
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.model = get_video_model(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from our DataLoader is (B, T, C, H, W).
        # The r2plus1d_18 model expects (B, C, T, H, W).
        # We permute the dimensions to match the model's expectation.
        x = x.permute(0, 2, 1, 3, 4)
        
        # Pass through the powerful pre-trained model
        return self.model(x).squeeze(1)