import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import IMAGENET_MEAN, IMAGENET_STD
from models import build_resnet
from utils import ensure_dir, save_json, set_seed


DEFAULT_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_CHOICES = ["resnet18", "resnet50", "resnet101"]


def configure_plot_style():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]


def save_history_csv(history, output_path):
    if not history:
        return
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history, output_path, title):
    if not history:
        return

    configure_plot_style()
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="训练损失")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="验证损失")
    axes[0].set_title("Loss 曲线")
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="训练准确率")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="验证准确率")
    axes[1].set_title("Accuracy 曲线")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [row["train_macro_f1"] for row in history], marker="o", label="训练F1")
    axes[2].plot(epochs, [row["val_macro_f1"] for row in history], marker="o", label="验证F1")
    axes[2].set_title("F1 曲线")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_mean_history(fold_results):
    histories = [result["history"] for result in fold_results if result.get("history")]
    if not histories:
        return []

    max_epoch = max(len(history) for history in histories)
    mean_history = []
    metric_keys = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
    ]

    for epoch_idx in range(max_epoch):
        row = {"epoch": epoch_idx + 1}
        for key in metric_keys:
            values = [history[epoch_idx][key] for history in histories if epoch_idx < len(history)]
            row[key] = float(np.mean(values)) if values else 0.0
        mean_history.append(row)
    return mean_history


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet 皮肤病变分类训练主程序。")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "all_data")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "HAM10000_metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "model_save")
    parser.add_argument("--architecture", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--all-resnets", action="store_true", help="依次训练 resnet18、resnet50、resnet101。")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--classifier-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    args.use_discriminative_lr = args.backbone_learning_rate is not None or args.classifier_learning_rate is not None
    if args.use_discriminative_lr:
        if args.classifier_learning_rate is None:
            args.classifier_learning_rate = args.learning_rate
        if args.backbone_learning_rate is None:
            args.backbone_learning_rate = args.classifier_learning_rate * 0.2
    else:
        args.backbone_learning_rate = args.learning_rate
        args.classifier_learning_rate = args.learning_rate
    return args


def select_architectures(args):
    if args.all_resnets:
        return MODEL_CHOICES
    if args.architecture:
        return [args.architecture]

    print("\n请选择要训练的模型：")
    print("1. ResNet18")
    print("2. ResNet50")
    print("3. ResNet101")
    print("4. ResNet18 + ResNet50 + ResNet101 全部训练")

    try:
        choice = input("请输入序号 [默认 2]: ").strip()
    except EOFError:
        choice = "2"

    if choice == "1":
        return ["resnet18"]
    if choice == "3":
        return ["resnet101"]
    if choice == "4":
        return MODEL_CHOICES
    return ["resnet50"]


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def read_lesion_groups(metadata_path, image_paths):
    if not metadata_path.exists():
        print(f"未找到元数据文件: {metadata_path}")
        print("已关闭病灶分组划分，回退为普通分层 K 折。")
        return None

    image_to_lesion = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_to_lesion[row["image_id"]] = row["lesion_id"]

    groups = []
    missing = 0
    for path, _ in image_paths:
        image_id = Path(path).stem.replace("train_", "")
        lesion_id = image_to_lesion.get(image_id)
        if lesion_id is None:
            missing += 1
            lesion_id = image_id
        groups.append(lesion_id)

    if missing:
        print(f"警告：{missing} 张图片缺少 lesion_id 元数据，已改用 image_id 作为分组。")
    return groups


def build_sampler(dataset):
    if isinstance(dataset, TransformSubset) and isinstance(dataset.subset, Subset):
        source_dataset = dataset.subset.dataset
        indices = dataset.subset.indices
        if hasattr(source_dataset, "targets"):
            labels = [source_dataset.targets[int(i)] for i in indices]
        else:
            labels = [source_dataset.samples[int(i)][1] for i in indices]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def add_dropout_to_classifier(model, dropout):
    if dropout <= 0:
        return model
    original_fc = model.fc
    model.fc = nn.Sequential(nn.Dropout(p=dropout), original_fc)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("fc.")


def build_optimizer(model, args):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not args.use_discriminative_lr:
        return optim.AdamW(
            [{"params": trainable_params, "lr": args.learning_rate, "name": "unified"}],
            weight_decay=args.weight_decay,
        )

    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.backbone_learning_rate,
                "name": "backbone",
            }
        )
    if classifier_params:
        param_groups.append(
            {
                "params": classifier_params,
                "lr": args.classifier_learning_rate,
                "name": "classifier",
            }
        )

    return optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )


def get_group_learning_rates(optimizer):
    rates = {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
    if "unified" in rates:
        return {
            "backbone": rates["unified"],
            "classifier": rates["unified"],
        }
    return {
        "backbone": rates.get("backbone", 0.0),
        "classifier": rates.get("classifier", rates.get("group_0", 0.0)),
    }


class TopKCheckpointManager:
    def __init__(self, save_dir, fold_idx, top_k):
        self.save_dir = ensure_dir(save_dir)
        self.fold_idx = fold_idx
        self.top_k = top_k
        self.records = []

    def update(self, model, metric, epoch):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fold{self.fold_idx}_epoch{epoch}_f1_{metric:.4f}_{run_id}.pth"
        save_path = self.save_dir / filename
        torch.save(model.state_dict(), save_path)
        self.records.append({"metric": metric, "path": save_path})
        self.records.sort(key=lambda item: item["metric"], reverse=True)

        for stale in self.records[self.top_k :]:
            if stale["path"].exists():
                stale["path"].unlink()
        self.records = self.records[: self.top_k]


def train_one_fold(model, train_loader, val_loader, args, save_dir, fold_idx, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    checkpoint_manager = TopKCheckpointManager(save_dir, fold_idx, args.top_k)

    if args.freeze_epochs > 0:
        set_backbone_trainable(model, trainable=False)
        print(f"第 {fold_idx} 折：前 {args.freeze_epochs} 轮冻结 ResNet 主干，只训练分类层。")
    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.time()
    best_model_path = save_dir / f"fold{fold_idx}_best_model.pth"

    print(f"\n第 {fold_idx} 折：开始训练")
    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, args)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            print(f"第 {fold_idx} 折：第 {epoch} 轮开始解冻全部网络。")

        model.train()
        train_loss = 0.0
        train_labels = []
        train_preds = []

        for inputs, labels in tqdm(train_loader, desc=f"第 {fold_idx} 折 第 {epoch} 轮 训练", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_labels.extend(labels.cpu().numpy().tolist())
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())

        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"第 {fold_idx} 折 第 {epoch} 轮 验证", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_labels.extend(labels.cpu().numpy().tolist())
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())

        train_metrics = compute_metrics(train_labels, train_preds)
        val_metrics = compute_metrics(val_labels, val_preds)
        scheduler.step(val_metrics["macro_f1"])
        current_lrs = get_group_learning_rates(optimizer)
        row = {
            "epoch": epoch,
            "backbone_learning_rate": current_lrs["backbone"],
            "classifier_learning_rate": current_lrs["classifier"],
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val_loss / len(val_loader.dataset),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        print(
            f"第 {fold_idx} 折 第 {epoch:03d} 轮 | "
            f"训练F1={train_metrics['macro_f1']:.4f} "
            f"验证F1={val_metrics['macro_f1']:.4f} "
            f"验证准确率={val_metrics['accuracy']:.4f} "
            f"主干LR={current_lrs['backbone']:.2e} "
            f"分类层LR={current_lrs['classifier']:.2e}"
        )

        checkpoint_manager.update(model, val_metrics["macro_f1"], epoch)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"第 {fold_idx} 折：第 {epoch} 轮触发早停。")
                break

    elapsed = time.time() - start_time
    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_model_path": str(best_model_path.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": elapsed,
        "history": history,
    }


def run_training_for_architecture(architecture, args, base_dataset, labels, groups, class_names, device):
    train_transform, eval_transform = build_transforms(args.image_size)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = ensure_dir(args.output_dir / architecture / run_time)
    use_persistent_workers = args.num_workers > 0
    use_pin_memory = device.type == "cuda"

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(np.zeros(len(labels)), labels, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(np.zeros(len(labels)), labels)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_subset = TransformSubset(Subset(base_dataset, train_idx), transform=train_transform)
        val_subset = TransformSubset(Subset(base_dataset, val_idx), transform=eval_transform)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=build_sampler(train_subset),
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent_workers,
        )

        model = build_resnet(
            architecture,
            num_classes=len(class_names),
            pretrained=not args.no_pretrained,
        )
        model = add_dropout_to_classifier(model, args.dropout).to(device)

        result = train_one_fold(model, train_loader, val_loader, args, save_dir, fold_idx, device)
        history_csv = save_dir / f"fold{fold_idx}_history.csv"
        curve_png = save_dir / f"fold{fold_idx}_curves.png"
        save_history_csv(result["history"], history_csv)
        plot_training_curves(result["history"], curve_png, f"{architecture} 第 {fold_idx} 折训练曲线")
        result["history_csv"] = str(history_csv.relative_to(PROJECT_ROOT))
        result["curve_png"] = str(curve_png.relative_to(PROJECT_ROOT))
        fold_results.append(result)

    mean_history = build_mean_history(fold_results)
    mean_history_csv = save_dir / "mean_history.csv"
    mean_curve_png = save_dir / "mean_curves.png"
    save_history_csv(mean_history, mean_history_csv)
    plot_training_curves(mean_history, mean_curve_png, f"{architecture} 平均训练曲线")

    summary = {
        "architecture": architecture,
        "class_names": class_names,
        "folds": args.folds,
        "save_dir": str(save_dir.relative_to(PROJECT_ROOT)),
        "regularization": {
            "label_smoothing": args.label_smoothing,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "use_discriminative_lr": args.use_discriminative_lr,
            "backbone_learning_rate": args.backbone_learning_rate,
            "classifier_learning_rate": args.classifier_learning_rate,
            "freeze_epochs": args.freeze_epochs,
        },
        "mean_val_macro_f1": float(np.mean([item["best_val_macro_f1"] for item in fold_results])),
        "std_val_macro_f1": float(np.std([item["best_val_macro_f1"] for item in fold_results])),
        "mean_history_csv": str(mean_history_csv.relative_to(PROJECT_ROOT)),
        "mean_curve_png": str(mean_curve_png.relative_to(PROJECT_ROOT)),
        "fold_results": fold_results,
    }
    save_json(summary, save_dir / "summary.json")
    print(
        f"\n{architecture} 训练完成 | 平均F1={summary['mean_val_macro_f1']:.4f} "
        f"+/- {summary['std_val_macro_f1']:.4f}"
    )
    print(f"训练结果已保存至: {save_dir}")
    print(f"平均曲线图: {mean_curve_png}")
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录: {args.data_dir}")

    base_dataset = datasets.ImageFolder(args.data_dir)
    labels = [label for _, label in base_dataset.samples]
    class_names = base_dataset.classes or DEFAULT_CLASS_NAMES
    groups = read_lesion_groups(args.metadata, base_dataset.samples)
    architectures = select_architectures(args)

    print(f"运行设备: {device}")
    print(f"数据集目录: {args.data_dir}")
    print(f"图片数量: {len(base_dataset)}")
    print(f"类别列表: {class_names}")
    print(f"病灶分组划分: {'已启用' if groups is not None else '未启用'}")
    print(f"本次训练模型: {architectures}")
    print(
        "抗过拟合设置: "
        f"label_smoothing={args.label_smoothing}, "
        f"dropout={args.dropout}, "
        f"weight_decay={args.weight_decay}, "
        f"learning_rate={args.learning_rate}, "
        f"use_discriminative_lr={args.use_discriminative_lr}, "
        f"freeze_epochs={args.freeze_epochs}"
    )

    summaries = [
        run_training_for_architecture(architecture, args, base_dataset, labels, groups, class_names, device)
        for architecture in architectures
    ]
    latest_summary_path = ensure_dir(PROJECT_ROOT / "outputs" / "runs") / "latest_train_main_summary.json"
    save_json(
        {
            "architectures": architectures,
            "summaries": summaries,
        },
        latest_summary_path,
    )
    print(f"最新训练总览已保存至: {latest_summary_path}")


if __name__ == "__main__":
    main()
