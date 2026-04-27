# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os
import glob
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import numpy as np
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

# ================= Configuration =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = str(PROJECT_ROOT / 'model_save' / '20260412_1215' / 'resnet')
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names mapping
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES_CN = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

# Threshold strategy configuration
NV_SUPPRESSION_THRESHOLD = 0.5
PRIORITY_THRESHOLDS = [
    ('mel', 0.25),
    ('bcc', 0.35),
    ('akiec', 0.30)
]

# Global model list
models_list = []
models_with_features_list = []
fc_weights_list = []
model_paths = []

# Image preprocessing
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_models():
    """Load all models"""
    global models_list, models_with_features_list, fc_weights_list, model_paths
    
    search_path = os.path.join(MODEL_DIR, "*.pth")
    model_paths = glob.glob(search_path)
    model_paths.sort()
    
    if len(model_paths) == 0:
        raise FileNotFoundError("No .pth model files found in {}!".format(MODEL_DIR))
    
    print("Loading {} models...".format(len(model_paths)))
    
    for path in model_paths:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        
        # Load weights
        state_dict = torch.load(path, map_location=DEVICE)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Handle DataParallel prefix
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        model = model.to(DEVICE)
        model.eval()
        models_list.append(model)
        
        # Save FC weights for heatmap generation
        fc_weights_list.append(state_dict['fc.weight'].float())
        
        # Create model with feature extraction
        model_with_features = ResNetWithFeatures(model)
        model_with_features.eval()
        models_with_features_list.append(model_with_features)
        
        print("  Loaded: {}".format(os.path.basename(path)))
    
    print("Model loading complete! Total {} models".format(len(models_list)))

class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model):
        super(ResNetWithFeatures, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc
    def forward(self, x):
        feature_maps = self.features(x)
        x = self.avgpool(feature_maps)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, feature_maps

def generate_heatmap(model_with_features, image_tensor, target_class, fc_weights):
    """Generate Grad-CAM heatmap using feature maps and FC weights"""
    model_with_features.eval()
    
    with torch.no_grad():
        logits, feature_maps = model_with_features(image_tensor)
        probs = F.softmax(logits, dim=1)
    
    _, batch_probs = torch.max(probs, dim=1)
    
    if target_class != batch_probs.item():
        target_class = batch_probs.item()
    
    # Get the feature maps (after global average pooling but before FC)
    # feature_maps shape: [1, 2048, 7, 7]
    
    # Get the FC weights for the target class
    # fc_weights shape: [7, 2048]
    class_weights = fc_weights[target_class]  # [2048]
    
    # Reshape weights to [2048, 1, 1] for broadcasting
    class_weights = class_weights.view(2048, 1, 1)
    
    # Compute weighted sum of feature maps
    # feature_maps: [1, 2048, 7, 7]
    # class_weights: [2048, 1, 1]
    # result: [1, 7, 7]
    cam = torch.sum(feature_maps * class_weights, dim=1).squeeze()
    
    # Apply ReLU
    cam = torch.relu(cam)
    
    # Normalize to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    
    # Resize to 224x224
    from torchvision.transforms import functional as TF
    cam = TF.resize(cam.unsqueeze(0), (224, 224)).squeeze()
    
    return cam.cpu().numpy()

def ensemble_predict(image):
    """Ensemble inference with heatmap"""
    # Preprocess image
    image_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    
    # Get NV index
    try:
        nv_idx = CLASS_NAMES.index('nv')
    except ValueError:
        nv_idx = -1
    
    # Parse threshold configs
    threshold_configs = []
    for cls_name, threshold in PRIORITY_THRESHOLDS:
        try:
            idx = CLASS_NAMES.index(cls_name)
            threshold_configs.append({'idx': idx, 'name': cls_name, 'thresh': threshold})
        except ValueError:
            pass
    
    # Multi-model prediction accumulation (without gradients)
    with torch.no_grad():
        probs_sum = torch.zeros(1, NUM_CLASSES).to(DEVICE)
        single_model_results = []
        
        for i, model in enumerate(models_list):
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            probs_sum += probs
            
            # Single model prediction result
            _, pred = torch.max(probs, 1)
            pred_class = CLASS_NAMES[pred.item()]
            single_model_results.append({
                'model': 'model_{}'.format(i+1),
                'prediction': pred_class,
                'prediction_cn': CLASS_NAMES_CN[pred_class],
                'confidence': probs[0][pred.item()].item(),
                'probabilities': {cls: probs[0][j].item() for j, cls in enumerate(CLASS_NAMES)}
            })
        
        # Average probabilities
        avg_probs = probs_sum / len(models_list)
        
        # Threshold correction strategy (direct prediction)
        _, base_pred = torch.max(avg_probs, 1)
        thresh_pred_idx = base_pred.clone()
        for config in reversed(threshold_configs):
            target_idx = config['idx']
            thresh_val = config['thresh']
            target_name = config['name']
            
            target_probs = avg_probs[:, target_idx]
            
            if target_name == 'mel' and nv_idx != -1:
                nv_probs = avg_probs[:, nv_idx]
                mask = (target_probs > thresh_val) & (nv_probs < NV_SUPPRESSION_THRESHOLD)
            else:
                mask = target_probs > thresh_val
            
            thresh_pred_idx[mask] = target_idx
        
        final_pred = CLASS_NAMES[thresh_pred_idx.item()]
        final_pred_idx = thresh_pred_idx.item()
        
        # Build result
        ensemble_probs = {cls: avg_probs[0][j].item() for j, cls in enumerate(CLASS_NAMES)}
    
    # Generate heatmap using the first model (with gradients)
    heatmap = generate_heatmap(models_with_features_list[0], image_tensor, final_pred_idx, fc_weights_list[0])
    
    # Convert heatmap to base64
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create heatmap overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    result = {
        'prediction': {
            'class': final_pred,
            'class_cn': CLASS_NAMES_CN[final_pred],
            'confidence': avg_probs[0][final_pred_idx].item()
        },
        'all_probabilities': {cls: round(prob, 4) for cls, prob in ensemble_probs.items()},
        'all_probabilities_cn': {CLASS_NAMES_CN[cls]: round(prob, 4) for cls, prob in ensemble_probs.items()},
        'single_model_predictions': single_model_results,
        'model_count': len(models_list),
        'heatmap_base64': heatmap_base64
    }
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Perform prediction
        result = ensemble_predict(image)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        result['image_base64'] = img_base64
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get loaded model information"""
    return jsonify({
        'model_count': len(models_list),
        'models': [os.path.basename(p) for p in model_paths],
        'device': str(DEVICE),
        'class_names': CLASS_NAMES,
        'class_names_cn': CLASS_NAMES_CN
    })

if __name__ == '__main__':
    # Load models
    load_models()
    
    # Start server
    port = 5001
    print(f"Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
