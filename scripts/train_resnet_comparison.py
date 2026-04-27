import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from skin_lesion.data import build_imagefolder_loaders
from skin_lesion.engine import evaluate_classifier, train_one_epoch
from skin_lesion.models import build_resnet
from skin_lesion.utils import ensure_dir, save_json, set_seed


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_architecture(architecture: str, config: dict, run_dir: Path, device: torch.device) -> dict:
    data_dir = PROJECT_ROOT / config.get("data_dir", "data")
    _, loaders = build_imagefolder_loaders(
        data_dir=data_dir,
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        seed=config.get("seed", 42),
        val_size=config.get("val_size", 0.2),
    )

    model = build_resnet(
        architecture,
        num_classes=config.get("num_classes", 7),
        pretrained=config.get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    arch_dir = ensure_dir(run_dir / architecture)
    best_f1 = -1.0
    history = []
    best_path = arch_dir / "best_model.pth"

    for epoch in range(1, config.get("epochs", 25) + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_classifier(model, loaders["val"], criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[{architecture}] epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "architecture": architecture,
                    "epoch": epoch,
                    "macro_f1": best_f1,
                    "state_dict": model.state_dict(),
                    "config": config,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_classifier(model, loaders["test"], criterion, device)

    result = {
        "architecture": architecture,
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["macro_f1"],
        "test_metrics": test_metrics,
        "best_model": str(best_path.relative_to(PROJECT_ROOT)),
        "history": history,
    }
    save_json(result, arch_dir / "metrics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18/50/101 comparison experiments.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "resnet_comparison.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(PROJECT_ROOT / config.get("output_dir", "outputs/runs") / f"resnet_comparison_{timestamp}")
    save_json(config, run_dir / "config.json")

    results = []
    for architecture in config.get("architectures", ["resnet18", "resnet50", "resnet101"]):
        results.append(train_architecture(architecture, config, run_dir, device))

    summary = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "results": results,
    }
    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
