import torch
import torch.nn as nn
from torchvision import models
import os
import glob
import numpy as np
from pathlib import Path

NUM_CLASSES = 7
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = str(PROJECT_ROOT / 'model_save' / '20260125_1112')
CPP_MODEL_DIR = str(PROJECT_ROOT / 'outputs' / 'exports' / 'onnx')

class ResNetWithFeatures(nn.Module):
    # 保持你之前的封装不变
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

def batch_export_ensemble():
    os.makedirs(CPP_MODEL_DIR, exist_ok=True)
    pth_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")))
    
    for i, pth in enumerate(pth_files):
        print(f"正在转换模型 {i+1}/9: {os.path.basename(pth)}")
        
        # 1. 初始化模型并加载权重
        base_model = models.resnet50(weights=None)
        base_model.fc = nn.Linear(base_model.fc.in_features, NUM_CLASSES)
        state_dict = torch.load(pth, map_location='cpu')
        state_dict = state_dict.get('state_dict', state_dict)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict, strict=True)
        
        # 2. 导出 FC 层权重 (.bin) 用于 C++ Grad-CAM
        fc_weights = state_dict['fc.weight'].float().numpy()
        bin_path = os.path.join(CPP_MODEL_DIR, f"resnet_fc_weights_{i}.bin")
        with open(bin_path, "wb") as f:
            f.write(fc_weights.tobytes())
            
        # 3. 导出 ONNX
        deploy_model = ResNetWithFeatures(base_model)
        deploy_model.eval()
        onnx_path = os.path.join(CPP_MODEL_DIR, f"skin_disease_resnet50_{i}.onnx")
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            deploy_model, dummy_input, onnx_path, export_params=True, 
            opset_version=12, do_constant_folding=True,
            input_names=['input'], output_names=['output', 'feature_maps']
        )
        
if __name__ == '__main__':
    batch_export_ensemble()
