# Skin Lesion Classification Project

这是毕业设计项目的整理版结构，保留了现有数据集、模型权重、Web 演示和论文图，同时把可复用代码放到 `src/skin_lesion` 中，方便后续做 ResNet18、ResNet50、ResNet101 对比实验。

## 目录结构

```text
.
├── configs/                 # 实验配置
├── data/                    # HAM10000 数据及划分结果
├── model_save/              # 已训练模型权重
├── notebooks/               # 原始 Notebook
├── outputs/                 # 新实验输出
├── reports/figures/         # 论文图表
├── scripts/                 # 可直接运行的脚本入口
├── src/skin_lesion/         # 训练、模型、数据加载等复用代码
└── web_app/                 # Flask Web 演示
```

## 常用命令

安装依赖后，可以从项目根目录运行：

```bash
pip install -r requirements.txt
python scripts/train_resnet_comparison.py --config configs/resnet_comparison.yaml
python scripts/evaluate.py
python web_app/app.py
```

`scripts/train_resnet_comparison.py` 会按配置依次训练 `resnet18`、`resnet50`、`resnet101`，每个模型会保存最佳权重和指标 JSON，适合后续写论文对比表。

当前数据目录可以兼容两种形式：

```text
data/train + data/val + data/test
data/all_data + data/test
```

如果只有 `data/all_data`，训练脚本会用固定随机种子从中划分训练集和验证集。
