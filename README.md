# 基于ResNet50与Grad-CAM的皮肤病辅助诊断系统设计与实现

这是毕业设计项目的整理版结构。项目保留现有数据集、模型权重、Web 演示和论文图表，并把训练代码整理为可直接运行的 Python 脚本。

## 目录结构

```text
.
├── configs/                 # 实验配置
├── data/                    # HAM10000 数据及划分结果
├── model_save/              # 已训练模型权重和新训练结果
├── outputs/                 # 新实验总览输出
├── reports/figures/         # 论文图表
├── scripts/                 # 可直接运行的脚本入口
├── src/                   # 模型、数据加载、训练工具
└── web_app/                 # Flask Web 演示
```

## 主训练程序

直接运行主程序后，会进入模型选择阶段：

```bash
python scripts/train_main.py
```

可选项：

```text
1. ResNet18
2. ResNet50
3. ResNet101
4. ResNet18 + ResNet50 + ResNet101 全部训练
```

如果仍想用命令行指定模型，也可以：

```bash
python scripts/train_main.py --architecture resnet50
python scripts/train_main.py --all-resnets
```

## 抗过拟合设置

主程序已经加入默认抗过拟合策略：

- 更强数据增强：随机裁剪、旋转、平移缩放、颜色扰动、Random Erasing
- `label_smoothing=0.1`
- 分类层前加入 `dropout=0.3`
- `weight_decay=5e-4`
- 前 5 轮冻结 ResNet 主干，只训练分类层
- 默认使用统一学习率 `8e-5`
- 验证 F1 连续不提升时早停
- 根据验证 F1 自动降低学习率

这些参数都可以在命令行覆盖，例如：

```bash
python scripts/train_main.py --architecture resnet50 --dropout 0.3 --freeze-epochs 5
```

如需单独做分层学习率实验，可以手动设置主干和分类层学习率：

```bash
python scripts/train_main.py --architecture resnet50 --backbone-learning-rate 2e-5 --classifier-learning-rate 1e-4
```

## 结果保存

训练结果会按模型分别保存，例如：

```text
model_save/resnet18/20260427_130100/
model_save/resnet50/20260427_130100/
model_save/resnet101/20260427_130100/
```

每个结果目录包含：

- `fold*_history.csv`：每一折每轮训练指标
- `fold*_curves.png`：每一折的 loss、accuracy、F1 曲线
- `fold*_best_model.pth`：每一折最佳模型
- `mean_history.csv`：多折平均指标
- `mean_curves.png`：多折平均曲线
- `summary.json`：该模型完整训练结果

最新一次训练总览会额外保存到：

```text
outputs/runs/latest_train_main_summary.json
```

## 其他命令

安装依赖：

```bash
pip install -r requirements.txt
```

运行评估脚本：

```bash
python scripts/evaluate.py
```

启动 Web 演示：

```bash
python web_app/app.py
```

## 数据目录

当前训练脚本默认读取：

```text
data/all_data
```

如果元数据文件存在：

```text
data/HAM10000_metadata.csv
```

脚本会优先使用 `lesion_id` 做分组交叉验证，减少同一病灶图片泄漏到训练集和验证集两边的风险。
