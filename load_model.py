import torch
from model_base import UpSimpleCNN
from model_base2 import CustomCNN

# Set up device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Initialize the model
model = CustomCNN()
model = model.to(device)

# Load model weights
state_dict = torch.load("runs/Road-Sign-Classifier-5/best_checkpoint.pth")
model.load_state_dict(state_dict['model_state'])
model.eval()

# Create dummy input matching the expected input size for the model
# Shape must match the input dimensions of your CNN
dummy_input = torch.randn(1, 3, 180, 180, device=device)  # Adjust shape as per your model's expected input
print(f"Dummy input shape: {dummy_input.shape}")

# Convert the model to ONNX
onnx_export_path = "road_model.onnx"
torch.onnx.export(
    model,                             # Model being run
    (dummy_input),                    # Model input (or a tuple for multiple inputs)
    onnx_export_path,                  # Output file name
    export_params=True,                # Store the trained model parameters
    opset_version=11,                  # The ONNX opset version (11 is widely supported)
    do_constant_folding=True,          # Whether to execute constant folding for optimization
    input_names=["input"],            # The model's input name
    output_names=["output"],          # The model's output name
    dynamic_axes={"input": {0: "batch_size"}}  # Handle dynamic batch size if needed
)

print(f"ONNX model has been exported to: {onnx_export_path}")
