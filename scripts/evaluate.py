import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import CLASS_NAMES, DEFAULT_NV_SUPPRESSION_THRESHOLD, DEFAULT_PRIORITY_THRESHOLDS
from data import build_eval_transform
from models import load_resnet_checkpoint
from utils import ensure_dir, set_seed

ARCHITECTURES = ("resnet18", "resnet50", "resnet101")

CLASS_NAME_ZH = {
    "akiec": "光化性角化病",
    "bcc": "基底细胞癌",
    "bkl": "良性角化病",
    "df": "皮肤纤维瘤",
    "mel": "黑色素瘤",
    "nv": "黑色素细胞痣",
    "vasc": "血管性病变",
}

COLUMN_NAME_ZH = {
    "strategy": "评估策略",
    "fold": "折次",
    "checkpoint": "模型文件",
    "checkpoint_val_f1": "训练验证F1",
    "accuracy": "准确率",
    "macro_precision": "宏平均精确率",
    "macro_recall": "宏平均召回率",
    "macro_f1": "宏平均F1",
    "weighted_f1": "加权F1",
    "macro_ovr_auc": "宏平均OvR AUC",
    "weighted_ovr_auc": "加权OvR AUC",
    "class": "类别",
    "precision": "精确率",
    "recall": "召回率",
    "f1-score": "F1分数",
    "support": "样本数",
}

ROW_NAME_ZH = {
    "accuracy": "准确率",
    "macro avg": "宏平均",
    "weighted avg": "加权平均",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估单个 ResNet 架构的三折集成模型。")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--run-dir", type=Path, default=None, help="包含 fold*_best_model.pth 的目录。")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluate")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_chinese_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]


def class_label(name: str) -> str:
    return CLASS_NAME_ZH.get(name, name)


def translate_value(value):
    if isinstance(value, str):
        return ROW_NAME_ZH.get(value, CLASS_NAME_ZH.get(value, value))
    return value


def table_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(columns=COLUMN_NAME_ZH)
    return out.apply(lambda column: column.map(translate_value))


def format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[format_cell(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, path_base: Path) -> None:
    path_base.with_suffix(".md").write_text(to_markdown_table(table_for_output(df)), encoding="utf-8")


def find_latest_complete_run(architecture: str) -> Path:
    root = PROJECT_ROOT / "model_save" / architecture
    candidates = sorted([p for p in root.glob("*") if p.is_dir()], reverse=True)
    for candidate in candidates:
        if all((candidate / f"fold{i}_best_model.pth").exists() for i in range(1, 4)):
            return candidate
    raise FileNotFoundError(f"在 {root} 下没有找到完整的三折模型目录。")


def load_fold_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / f"fold{i}_best_model.pth" for i in range(1, 4)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少模型文件: " + ", ".join(missing))
    return paths


def checkpoint_score(path: Path) -> float:
    match = re.search(r"_f1_([0-9.]+)_", path.name)
    return float(match.group(1)) if match else float("nan")


def predict_ensemble(
    architecture: str,
    fold_paths: list[Path],
    loader: DataLoader,
    class_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y_true_batches = []
    ensemble_batches = []
    fold_prob_batches = [[] for _ in fold_paths]

    models = [
        load_resnet_checkpoint(path, architecture=architecture, num_classes=class_count, device=device)
        for path in fold_paths
    ]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probs = [F.softmax(model(inputs), dim=1).cpu().numpy() for model in models]
            for idx, fold_probs in enumerate(probs):
                fold_prob_batches[idx].append(fold_probs)
            ensemble_batches.append(np.stack(probs, axis=0).mean(axis=0))
            y_true_batches.append(labels.numpy())

    y_true = np.concatenate(y_true_batches)
    y_prob = np.concatenate(ensemble_batches)
    fold_probs = [np.concatenate(items) for items in fold_prob_batches]

    fold_rows = []
    for idx, (path, prob) in enumerate(zip(fold_paths, fold_probs), start=1):
        pred = np.argmax(prob, axis=1)
        fold_rows.append(
            {
                "fold": idx,
                "checkpoint": str(path.relative_to(PROJECT_ROOT)),
                "checkpoint_val_f1": checkpoint_score(path),
                "accuracy": accuracy_score(y_true, pred),
                "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
                "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
            }
        )
    return y_true, y_prob, fold_rows


def apply_priority_thresholds(y_prob: np.ndarray, class_names: list[str]) -> np.ndarray:
    preds = np.argmax(y_prob, axis=1)
    nv_idx = class_names.index("nv") if "nv" in class_names else None
    for class_name, threshold in reversed(DEFAULT_PRIORITY_THRESHOLDS):
        if class_name not in class_names:
            continue
        class_idx = class_names.index(class_name)
        mask = y_prob[:, class_idx] > threshold
        if class_name == "mel" and nv_idx is not None:
            mask = mask & (y_prob[:, nv_idx] < DEFAULT_NV_SUPPRESSION_THRESHOLD)
        preds[mask] = class_idx
    return preds


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "strategy": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_ovr_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        "weighted_ovr_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [class_label(name) for name in class_names]

    plt.figure(figsize=(8.5, 7.2))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("归一化混淆矩阵")
    plt.tight_layout()
    plt.savefig(out_dir / "归一化混淆矩阵.png", dpi=300)
    plt.close()


def save_roc_figure(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(8.5, 7.0))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        auc_value = roc_auc_score(y_bin[:, idx], y_prob[:, idx])
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{class_label(class_name)} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title("一对其余 ROC 曲线")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "ROC曲线.png", dpi=300)
    plt.close()


def save_melanoma_threshold_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    if "mel" not in class_names:
        return

    mel_idx = class_names.index("mel")
    y_mel = (y_true == mel_idx).astype(int)
    thresholds = np.linspace(0, 1, 101)
    rows = []
    for threshold in thresholds:
        pred = (y_prob[:, mel_idx] >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_mel, pred, zero_division=0),
                "recall": recall_score(y_mel, pred, zero_division=0),
                "f1": f1_score(y_mel, pred, zero_division=0),
            }
        )
    df = pd.DataFrame(rows)

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(df["threshold"], df["precision"], label="精确率", linewidth=2)
    plt.plot(df["threshold"], df["recall"], label="召回率", linewidth=2)
    plt.plot(df["threshold"], df["f1"], label="F1分数", linewidth=2)
    plt.axvline(0.25, color="black", linestyle="--", linewidth=1.2, label="选定阈值=0.25")
    plt.xlabel("黑色素瘤预测概率阈值")
    plt.ylabel("指标值")
    plt.title("黑色素瘤阈值敏感性分析")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "黑色素瘤阈值敏感性.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    sns.set_theme(style="whitegrid", context="paper")
    configure_chinese_style()

    run_dir = args.run_dir or find_latest_complete_run(args.architecture)
    out_dir = ensure_dir(args.output_dir)
    print(f"正在评估 {args.architecture}，结果将保存到 {out_dir}")

    dataset = datasets.ImageFolder(args.data_dir, transform=build_eval_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = dataset.classes or CLASS_NAMES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_paths = load_fold_paths(run_dir)

    y_true, y_prob, fold_rows = predict_ensemble(args.architecture, fold_paths, loader, len(class_names), device)
    pred_argmax = np.argmax(y_prob, axis=1)
    pred_threshold = apply_priority_thresholds(y_prob, class_names)

    metrics = pd.DataFrame(
        [
            metric_row("三折集成-最大概率策略", y_true, pred_argmax, y_prob),
            metric_row("三折集成-临床阈值策略", y_true, pred_threshold, y_prob),
        ]
    )
    report = pd.DataFrame(classification_report(y_true, pred_argmax, target_names=class_names, output_dict=True)).T

    save_table(metrics, out_dir / "总体指标")
    save_table(report.reset_index(names="class"), out_dir / "分类报告")
    save_table(pd.DataFrame(fold_rows), out_dir / "各折指标")
    save_confusion_matrix(y_true, pred_argmax, class_names, out_dir)
    save_roc_figure(y_true, y_prob, class_names, out_dir)
    save_melanoma_threshold_figure(y_true, y_prob, class_names, out_dir)

    print("评估图表已保存。")


if __name__ == "__main__":
    main()
