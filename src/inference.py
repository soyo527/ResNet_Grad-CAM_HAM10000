import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from constants import CLASS_NAMES, CLASS_NAMES_EN, IMAGENET_MEAN, IMAGENET_STD
from data import build_eval_transform
from models import load_resnet_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("resnet18", "resnet50", "resnet101")


def latest_complete_run(architecture: str) -> Path:
    root = PROJECT_ROOT / "model_save" / architecture
    candidates = sorted([path for path in root.glob("*") if path.is_dir()], reverse=True)
    for candidate in candidates:
        if all((candidate / f"fold{idx}_best_model.pth").exists() for idx in range(1, 4)):
            return candidate
    raise FileNotFoundError(f"No complete three-fold checkpoint directory found under {root}")


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_rgb_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


class SkinLesionPredictor:
    def __init__(self, architecture: str = "resnet50", image_size: int = 224):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.architecture = architecture
        self.image_size = image_size
        self.class_names = CLASS_NAMES
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_eval_transform(image_size)
        self.run_dir = latest_complete_run(architecture)
        self.checkpoint_paths = [self.run_dir / f"fold{idx}_best_model.pth" for idx in range(1, 4)]
        self.models = [
            load_resnet_checkpoint(path, architecture=architecture, num_classes=len(self.class_names), device=self.device)
            for path in self.checkpoint_paths
        ]

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fold_probs = [F.softmax(model(tensor), dim=1) for model in self.models]
            probabilities = torch.stack(fold_probs).mean(dim=0).squeeze(0).cpu().numpy()

        predicted_index = int(np.argmax(probabilities))
        heatmap = self.grad_cam(image, tensor, predicted_index)
        rows = [
            {
                "code": code,
                "name": CLASS_NAMES_EN.get(code, code),
                "probability": float(probabilities[idx]),
                "percent": round(float(probabilities[idx]) * 100, 2),
            }
            for idx, code in enumerate(self.class_names)
        ]
        rows.sort(key=lambda item: item["probability"], reverse=True)

        return {
            "architecture": self.architecture,
            "run_dir": str(self.run_dir.relative_to(PROJECT_ROOT)),
            "prediction": rows[0],
            "probabilities": rows,
            "heatmap": encode_png(heatmap),
        }

    def grad_cam(self, image: Image.Image, tensor: torch.Tensor, target_index: int) -> Image.Image:
        model = self.models[0]
        activations = []
        gradients = []

        def save_activation(_module, _inputs, output):
            activations.append(output)

        def save_gradient(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = model.layer4.register_forward_hook(save_activation)
        backward_handle = model.layer4.register_full_backward_hook(save_gradient)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            logits[:, target_index].sum().backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        cam_array = cam.cpu().numpy()

        heatmap = Image.fromarray(np.uint8(cam_array * 255), mode="L").resize(image.size, Image.Resampling.BILINEAR)
        heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
        colored = colormaps["jet"](heatmap_array)[..., :3]
        colored = Image.fromarray(np.uint8(colored * 255), mode="RGB")
        return Image.blend(image.convert("RGB"), colored, alpha=0.42)
