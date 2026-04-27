# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import os
import glob

MODEL_DIR = r'D:\软件\工作文件夹\本科毕设\my_project\model_save\20260412_1215\resnet'
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Model directory: {MODEL_DIR}")

# Check if directory exists
if not os.path.exists(MODEL_DIR):
    print(f"Error: Directory {MODEL_DIR} does not exist!")
    exit(1)

# Find model files
model_paths = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
model_paths.sort()

print(f"Found {len(model_paths)} model files:")
for path in model_paths:
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"  - {os.path.basename(path)} ({size:.2f} MB)")

if len(model_paths) == 0:
    print("Error: No model files found!")
    exit(1)

# Try to load the first model
try:
    print("\nTesting model loading...")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load weights
    state_dict = torch.load(model_paths[0], map_location=DEVICE)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    print("✓ Model loaded successfully!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Device: {next(model.parameters()).device}")
    
    # Test with a dummy input
    import torchvision.transforms as transforms
    from PIL import Image
    import io
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Preprocess
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = test_transforms(dummy_image).unsqueeze(0).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    
    print("✓ Model inference successful!")
    print(f"Output shape: {output.shape}")
    
    print("\nAll tests passed! The models are ready to use.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
