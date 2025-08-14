# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import VOCAB_SIZE from dataset.py
# If dataset.py hasn't been created yet when testing model.py standalone,
# provide a fallback default value.
try:
    from dataset import VOCAB_SIZE
except ImportError:
    print("Warning: Could not import VOCAB_SIZE from dataset.py. Using default value 37 for testing model.py.")
    # Fallback value based on GRID dataset charset defined in dataset.py (" abcdefghijklmnopqrstuvwxyz0123456789")
    VOCAB_SIZE = 37


class LipReadModel(nn.Module):
    """
    Defines the CNN + LSTM + CTC model for lip reading.
    Input: Batch of video sequences (B, T, C, H, W)
    Output: Log probabilities for CTC loss (T, B, N)
    """
    def __init__(self, input_channels=1, cnn_output_dim=512, lstm_hidden_dim=256,
                 lstm_layers=2, num_classes=VOCAB_SIZE, dropout_p=0.3):
        """
        Initializes the model layers.
        Args:
            input_channels (int): Number of channels in input images (1 for grayscale).
            cnn_output_dim (int): Dimension of features output by the CNN backbone before LSTM.
            lstm_hidden_dim (int): Hidden dimension size for each LSTM layer direction.
            lstm_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output classes (size of vocabulary including blank).
            dropout_p (float): Dropout probability.
        """
        super(LipReadModel, self).__init__()

        if num_classes <= 0:
             raise ValueError("num_classes must be positive. Check VOCAB_SIZE import.")

        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim

        # --- CNN Backbone (Spatial Feature Extractor) ---
        # Processes each frame. Input shape per frame: (B*T, C, H, W)
        # Output shape per frame: (B*T, cnn_output_dim)

        # Layer 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False) # Bias False often used with BatchNorm
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 96x96 -> 48x48
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 48x48 -> 24x24
        # Layer 3
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 24x24 -> 12x12
        # Layer 4
        self.conv4 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 12x12 -> 6x6

        # Calculate the flattened size after convolutions and pooling
        # Input H=96, W=96. After 4 pooling layers (divide by 2^4 = 16): 96/16 = 6
        # Final feature map size: 6x6 with 128 channels
        self.cnn_output_flat_size = 128 * 6 * 6 # 4608

        # Fully Connected layer after CNN pooling to project features
        self.cnn_fc = nn.Linear(self.cnn_output_flat_size, cnn_output_dim)
        self.cnn_dropout = nn.Dropout(dropout_p)

        # --- LSTM (Temporal Feature Aggregator) ---
        # Input shape: (B, T, cnn_output_dim)
        # Output shape: (B, T, lstm_hidden_dim * 2) because bidirectional=True
        self.lstm = nn.LSTM(input_size=cnn_output_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=True,   # Capture context from both past and future
                            batch_first=True,     # Input/output format: (Batch, Time, Features)
                            dropout=dropout_p if lstm_layers > 1 else 0) # Dropout between LSTM layers

        # --- Classifier ---
        # Maps LSTM output features to class logits for each time step
        # Input shape: (B, T, lstm_hidden_dim * 2)
        # Output shape: (B, T, num_classes)
        self.fc_classifier = nn.Linear(lstm_hidden_dim * 2, num_classes)
        # Note: No Softmax here, CTCLoss expects raw logits.

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (Tensor): Input batch of video sequences (B, T, C, H, W).
        Returns:
            Tensor: Log probabilities output, ready for CTC loss (T, B, N).
        """
        # Input x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()

        # --- CNN Pass ---
        # Reshape for CNN: Combine Batch and Time dimensions
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_cnn = x.view(B * T, C, H, W)

        # Apply CNN layers with ReLU activation and Pooling
        out = self.pool1(F.relu(self.bn1(self.conv1(x_cnn))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.pool4(F.relu(self.bn4(self.conv4(out)))) # Shape: (B*T, 128, 6, 6)

        # Flatten the output from CNN pooling layers
        out_flat = out.view(B * T, -1) # Shape: (B*T, 4608)

        # Apply fully connected layer and dropout after CNN backbone
        cnn_features = self.cnn_dropout(F.relu(self.cnn_fc(out_flat))) # Shape: (B*T, cnn_output_dim)

        # --- LSTM Pass ---
        # Reshape for LSTM: Separate Batch and Time dimensions again
        # (B*T, cnn_output_dim) -> (B, T, cnn_output_dim)
        lstm_input = cnn_features.view(B, T, -1)

        # Apply LSTM
        # Note: For efficiency with variable lengths (not needed for GRID's fixed length),
        # one might use pack_padded_sequence here based on input_lengths.
        # For simplicity with fixed length, we pass directly.
        # self.lstm.flatten_parameters() # Useful if using DataParallel or DDP
        lstm_out, _ = self.lstm(lstm_input) # lstm_out shape: (B, T, lstm_hidden_dim * 2)

        # --- Classifier Pass ---
        # Apply final linear layer to get logits for each time step
        logits = self.fc_classifier(lstm_out) # Logits shape: (B, T, num_classes)

        # --- Prepare Output for CTCLoss ---
        # CTCLoss requires input shape: (Time, Batch, Classes) or (T, B, N)
        # It also requires log probabilities.
        log_probs = F.log_softmax(logits, dim=2) # Apply log_softmax over the class dimension (N)

        # Permute dimensions: (B, T, N) -> (T, B, N)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs

# --- Example Usage Block (for testing model.py standalone) ---
if __name__ == '__main__':
    print("Testing LipReadModel Architecture...")

    # Define dummy parameters for testing
    test_batch_size = 4
    test_seq_len = 75 # Max sequence length (T) for GRID
    test_channels = 1
    test_height = 96
    test_width = 96
    # Use the VOCAB_SIZE defined/imported at the top
    test_num_classes = VOCAB_SIZE

    # Create a dummy input tensor
    dummy_input = torch.randn(test_batch_size, test_seq_len, test_channels, test_height, test_width)
    print(f"\nDummy Input Shape: {dummy_input.shape}") # Should be (B, T, C, H, W)

    # --- Instantiate the Model ---
    # Use parameters consistent with the training script defaults/definitions
    test_model = LipReadModel(
        input_channels=test_channels,
        cnn_output_dim=512,    # Match train.py config
        lstm_hidden_dim=256,   # Match train.py config
        lstm_layers=2,         # Match train.py config
        num_classes=test_num_classes,
        dropout_p=0.3          # Match train.py config
    )

    # --- Print Model Summary ---
    print("\nModel Architecture Summary:")
    print(test_model)

    # --- Count Trainable Parameters ---
    num_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params:,}")

    # --- Test Forward Pass ---
    print("\nTesting Forward Pass...")
    try:
        # Move model and data to a device if needed for test (CPU is fine here)
        # test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test_model.to(test_device)
        # dummy_input = dummy_input.to(test_device)

        test_model.eval() # Set to evaluation mode for testing forward pass
        with torch.no_grad(): # Disable gradients for testing pass
             output_log_probs = test_model(dummy_input)

        print(f"Output shape (log_probs for CTC): {output_log_probs.shape}")
        # Expected output shape: (T, B, N) = (Time, Batch, NumClasses)
        expected_shape = (test_seq_len, test_batch_size, test_num_classes)
        assert output_log_probs.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output_log_probs.shape}"
        print(f"Output shape matches expected: {expected_shape}")
        print("Forward pass successful!")

        # Optional: Check output values
        # print(f"Output Log Probs (sample): {output_log_probs[:, 0, :5]}") # Log probs for first batch item, first 5 classes

    except Exception as e:
        print(f"Error during forward pass test: {e}")
        import traceback
        traceback.print_exc()