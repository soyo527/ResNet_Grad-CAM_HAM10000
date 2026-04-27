import argparse
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
    "architecture": "模型结构",
    "run_dir": "模型目录",
    "fold": "折次",
    "checkpoint": "模型文件",
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

METRIC_NAME_ZH = {
    "accuracy": "准确率",
    "macro_precision": "宏平均精确率",
    "macro_recall": "宏平均召回率",
    "macro_f1": "宏平均F1",
    "macro_ovr_auc": "宏平均OvR AUC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接推理比较 ResNet18、ResNet50、ResNet101 的三折模型。")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "comparison")
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


def latest_complete_run(architecture: str) -> Path:
    root = PROJECT_ROOT / "model_save" / architecture
    for run_dir in sorted([p for p in root.glob("*") if p.is_dir()], reverse=True):
        if all((run_dir / f"fold{i}_best_model.pth").exists() for i in range(1, 4)):
            return run_dir
    raise FileNotFoundError(f"在 {root} 下没有找到完整的三折模型目录。")


def fold_paths(run_dir: Path) -> list[Path]:
    return [run_dir / f"fold{i}_best_model.pth" for i in range(1, 4)]


def infer_architecture(
    architecture: str,
    paths: list[Path],
    loader: DataLoader,
    class_count: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    fold_batches = []
    y_true_batches = []
    models = [load_resnet_checkpoint(path, architecture, class_count, device) for path in paths]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probs = [F.softmax(model(inputs), dim=1).cpu().numpy() for model in models]
            fold_batches.append(np.stack(probs, axis=0))
            y_true_batches.append(labels.numpy())

    stacked = np.concatenate(fold_batches, axis=1)
    return np.concatenate(y_true_batches), stacked.mean(axis=0), [stacked[i] for i in range(stacked.shape[0])]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_ovr_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        "weighted_ovr_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"),
    }


def save_overall_metric_figure(metrics: pd.DataFrame, out_dir: Path) -> None:
    plot_df = metrics.melt(
        id_vars=["architecture"],
        value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auc"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map(METRIC_NAME_ZH)

    plt.figure(figsize=(10.5, 5.8))
    sns.barplot(data=plot_df, x="metric", y="score", hue="architecture")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("指标值")
    plt.title("ResNet模型总体性能对比")
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="模型结构")
    plt.tight_layout()
    plt.savefig(out_dir / "总体指标柱状图.png", dpi=300)
    plt.close()


def save_nine_model_stability_figure(fold_metrics: pd.DataFrame, out_dir: Path) -> None:
    plot_df = fold_metrics.copy()
    plot_df["单模型"] = plot_df["architecture"] + "-fold" + plot_df["fold"].astype(str)

    plt.figure(figsize=(10.0, 5.8))
    ax = plt.gca()
    sns.boxplot(data=plot_df, x="architecture", y="macro_f1", color="#c6dbef", width=0.42)
    arch_to_x = {architecture: idx for idx, architecture in enumerate(ARCHITECTURES)}
    fold_offsets = {1: -0.14, 2: 0.0, 3: 0.14}
    fold_colors = {1: "#66c2a5", 2: "#fc8d62", 3: "#8da0cb"}
    for _, row in plot_df.iterrows():
        fold = int(row["fold"])
        x = arch_to_x[row["architecture"]] + fold_offsets[fold]
        y = row["macro_f1"]
        ax.scatter(x, y, s=70, color=fold_colors[fold], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x + 0.025, y, f"fold{fold}", fontsize=8, va="center")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=f"fold{fold}", markersize=7)
        for fold, color in fold_colors.items()
    ]
    plt.xlabel("模型结构")
    plt.ylabel("宏平均F1")
    plt.title("九个单模型的三折宏平均F1稳定性")
    plt.legend(handles=handles, title="折次", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "九个单模型三折稳定性.png", dpi=300)
    plt.close()


def save_roc_comparison(results: dict, class_names: list[str], out_dir: Path) -> None:
    y_true = next(iter(results.values()))["y_true"]
    y_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(8.0, 6.4))
    for architecture, item in results.items():
        auc_value = roc_auc_score(y_true, item["y_prob"], multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin.ravel(), item["y_prob"].ravel())
        plt.plot(fpr, tpr, linewidth=2, label=f"{architecture} 宏平均AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.title("宏平均ROC曲线对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "宏平均ROC对比.png", dpi=300)
    plt.close()


def save_class_heatmap(class_metrics: pd.DataFrame, out_dir: Path) -> None:
    heatmap_df = class_metrics.assign(class_name=class_metrics["class"].map(class_label))
    heatmap_df = heatmap_df.pivot(index="class_name", columns="architecture", values="f1-score")
    plt.figure(figsize=(6.8, 5.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.xlabel("模型结构")
    plt.ylabel("类别")
    plt.title("各类别F1分数对比")
    plt.tight_layout()
    plt.savefig(out_dir / "各类别F1热力图.png", dpi=300)
    plt.close()


def save_confusion_matrix_panel(results: dict, class_names: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    class_labels = [class_label(name) for name in class_names]
    for ax, (architecture, item) in zip(axes, results.items()):
        y_pred = np.argmax(item["y_prob"], axis=1)
        cm = confusion_matrix(item["y_true"], y_pred, labels=range(len(class_names)))
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        ax.set_title(architecture)
        ax.set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    fig.suptitle("归一化混淆矩阵对比")
    plt.tight_layout()
    plt.savefig(out_dir / "归一化混淆矩阵对比.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    sns.set_theme(style="whitegrid", context="paper")
    configure_chinese_style()
    out_dir = ensure_dir(args.output_dir)
    print(f"正在进行模型对比，结果将保存到 {out_dir}")

    dataset = datasets.ImageFolder(args.data_dir, transform=build_eval_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    fold_rows = []
    for architecture in ARCHITECTURES:
        run_dir = latest_complete_run(architecture)
        paths = fold_paths(run_dir)
        print(f"正在评估 {architecture}: {run_dir.relative_to(PROJECT_ROOT)}")
        y_true, y_prob, fold_prob_list = infer_architecture(architecture, paths, loader, len(class_names), device)
        results[architecture] = {
            "run_dir": run_dir,
            "checkpoints": paths,
            "y_true": y_true,
            "y_prob": y_prob,
        }
        for fold_idx, fold_prob in enumerate(fold_prob_list, start=1):
            row = {"architecture": architecture, "fold": fold_idx}
            row.update(compute_metrics(y_true, fold_prob))
            row["checkpoint"] = str(paths[fold_idx - 1].relative_to(PROJECT_ROOT))
            fold_rows.append(row)

    overall_rows = []
    class_rows = []
    for architecture, item in results.items():
        metrics = {"architecture": architecture, "run_dir": str(item["run_dir"].relative_to(PROJECT_ROOT))}
        metrics.update(compute_metrics(item["y_true"], item["y_prob"]))
        overall_rows.append(metrics)

        y_pred = np.argmax(item["y_prob"], axis=1)
        report = pd.DataFrame(
            classification_report(item["y_true"], y_pred, target_names=class_names, output_dict=True)
        ).T.reset_index(names="class")
        report.insert(0, "architecture", architecture)
        class_rows.append(report)

    overall_df = pd.DataFrame(overall_rows).sort_values("macro_f1", ascending=False)
    fold_df = pd.DataFrame(fold_rows)
    class_df = pd.concat(class_rows, ignore_index=True)
    class_df = class_df[class_df["class"].isin(class_names)]

    save_table(overall_df, out_dir / "总体指标")
    save_table(class_df, out_dir / "各类别指标")
    save_table(fold_df, out_dir / "九个单模型指标")
    save_overall_metric_figure(overall_df, out_dir)
    save_nine_model_stability_figure(fold_df, out_dir)
    save_roc_comparison(results, class_names, out_dir)
    save_class_heatmap(class_df, out_dir)
    save_confusion_matrix_panel(results, class_names, out_dir)

    print("对比图表已保存。")


if __name__ == "__main__":
    main()
