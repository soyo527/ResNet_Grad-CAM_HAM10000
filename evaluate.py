import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import glob
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# ================= 配置区域 =================
# 1. 路径配置
TEST_DIR = '/root/graduation_project/my_project/data/test' 
MODEL_DIR = '/root/graduation_project/my_project/model_save/20260125_1112'

# 2. 基础参数
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 阈值策略配置
# 这里的逻辑是：必须同时满足 (目标概率 > Threshold) AND (NV概率 < NV_LIMIT)
NV_SUPPRESSION_THRESHOLD = 0.5  # 【新增】NV 抑制阈值 (用户指定为 0.8)

PRIORITY_THRESHOLDS = [
    ('mel', 0.25),   # 第一优先级：黑色素瘤 (高危)
    ('bcc', 0.35),   # 第二优先级：基底细胞癌
    ('akiec', 0.30)  # 第三优先级：癌前病变
]

# 4. 绘图保存路径
PLOT_SAVE_DIR = '/root/graduation_project/my_project/thesis_plots'
# ===========================================

# 屏蔽警告
warnings.filterwarnings("ignore")

# ==========================================
# 🛠️ 工具函数：毕设绘图套件 (新增)
# ==========================================
def plot_thesis_figures(y_true, y_probs, single_f1_scores, class_names, save_dir=PLOT_SAVE_DIR):
    """一键生成毕设所需的 4 张核心图表"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 设置风格 (支持中文显示，如果环境不支持 SimHei 可改为 sans-serif)
    sns.set_style("whitegrid")
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] 
    except:
        pass
    
    n_classes = len(class_names)
    
    # --- 图 1: 多类别 ROC 曲线 ---
    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, '1_ROC_Curve.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ ROC 绘图失败: {e}")

    # --- 图 2: Mel 阈值敏感度分析 ---
    if 'mel' in class_names:
        mel_idx = class_names.index('mel')
        thresholds = np.arange(0, 1.01, 0.01)
        recalls, precisions, f1_scores = [], [], []
        
        y_true_mel = (np.array(y_true) == mel_idx).astype(int)
        mel_probs = y_probs[:, mel_idx]
        
        for t in thresholds:
            y_pred_t = (mel_probs > t).astype(int)
            tp = np.sum((y_pred_t == 1) & (y_true_mel == 1))
            fp = np.sum((y_pred_t == 1) & (y_true_mel == 0))
            fn = np.sum((y_pred_t == 0) & (y_true_mel == 1))
            
            r = tp / (tp + fn + 1e-7)
            p = tp / (tp + fp + 1e-7)
            f1 = 2 * p * r / (p + r + 1e-7)
            recalls.append(r); precisions.append(p); f1_scores.append(f1)
            
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, recalls, 'r-', label='Recall', lw=2)
        plt.plot(thresholds, precisions, 'b--', label='Precision', lw=2)
        plt.plot(thresholds, f1_scores, 'g-.', label='F1-Score', lw=2)
        plt.axvline(x=0.25, color='k', linestyle=':', label='Selected Threshold (0.25)')
        plt.xlabel('Threshold')
        plt.title('Melanoma Threshold Sensitivity Analysis')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '2_Threshold_Analysis.png'), dpi=300)
        plt.close()

    # --- 图 3: 归一化混淆矩阵 ---
    y_pred_argmax = np.argmax(y_probs, axis=1) 
    cm = confusion_matrix(y_true, y_pred_argmax)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, '3_Confusion_Matrix.png'), dpi=300)
    plt.close()

    # --- 图 4: K-Fold 稳定性箱线图 ---
    if single_f1_scores:
        plt.figure(figsize=(6, 6))
        sns.boxplot(y=single_f1_scores, color='skyblue', width=0.4)
        sns.stripplot(y=single_f1_scores, color='red', size=8, jitter=True, label='Single Model')
        plt.ylabel('Macro F1 Score')
        plt.title(f'Model Stability (Mean F1={np.mean(single_f1_scores):.3f})')
        plt.savefig(os.path.join(save_dir, '4_Stability_Boxplot.png'), dpi=300)
        plt.close()
    
    print(f"\n📊 [图表生成完毕] 所有图片已保存至: {save_dir}")

# ==========================================
# 核心加载函数
# ==========================================
def load_models_auto():
    search_path = os.path.join(MODEL_DIR, "*.pth")
    model_paths = glob.glob(search_path)
    model_paths.sort()
    
    if len(model_paths) == 0:
        # 尝试查找子文件夹
        subdirs = [d for d in glob.glob(os.path.join(MODEL_DIR, '*')) if os.path.isdir(d)]
        if subdirs:
            latest = max(subdirs, key=os.path.getctime)
            search_path = os.path.join(latest, "*.pth")
            model_paths = glob.glob(search_path)
            model_paths.sort()
    
    if len(model_paths) == 0:
        raise FileNotFoundError(f"在 {MODEL_DIR} 下没找到任何 .pth 模型文件！")
    
    models_list = []
    print(f"🚀 正在加载 {len(model_paths)} 个模型 (Top-K Ensemble)...")
    
    for path in model_paths:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        models_list.append(model)
        
    return models_list, model_paths

# ==========================================
# 主推理函数
# ==========================================
def ensemble_inference():
    # 1. 准备数据
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    real_test_dir = TEST_DIR
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ 没找到 {TEST_DIR}，临时使用 ./data/val 演示")
        real_test_dir = './data/val'
        
    try:
        dataset = datasets.ImageFolder(real_test_dir, transform=test_transforms)
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据目录 {real_test_dir}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    class_names = dataset.classes

    # 解析阈值配置
    threshold_configs = []
    print("\n⚖️  初始化风险阈值策略：")
    for cls_name, threshold in PRIORITY_THRESHOLDS:
        try:
            idx = class_names.index(cls_name)
            threshold_configs.append({'idx': idx, 'name': cls_name, 'thresh': threshold})
            print(f"   👉 优先级锁定: {cls_name:<6} (ID: {idx}) | 触发阈值 > {threshold}")
        except ValueError:
            pass
            
    # 查找 nv (黑色素痣) 的索引
    try:
        nv_idx = class_names.index('nv')
        print(f"   🛡️ 误报抑制开启: 当 nv > {NV_SUPPRESSION_THRESHOLD} 时，忽略 Mel 警报")
    except ValueError:
        nv_idx = -1
        print("   ⚠️ 警告: 未找到 'nv' 类别，误报抑制失效")

    # 2. 自动加载模型群
    models_list, model_paths = load_models_auto()
    
    y_true = []
    y_pred_ensemble = []   # 标准策略
    y_pred_thresh = []     # 阈值策略
    all_ensemble_probs = [] # 【新增】用于画 ROC 曲线
    
    single_model_preds = [[] for _ in range(len(models_list))]
    
    print("\n🚀 开始集成推理 (双策略并行计算)...")
    
    # 3. 推理循环
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            
            # 多模型预测累加
            batch_probs_sum = torch.zeros(inputs.size(0), NUM_CLASSES).to(DEVICE)
            for i, model in enumerate(models_list):
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                batch_probs_sum += probs
                
                # 记录单模型预测
                _, single_preds = torch.max(probs, 1)
                single_model_preds[i].extend(single_preds.cpu().numpy())
            
            # 取平均
            avg_probs = batch_probs_sum / len(models_list)
            
            # 【关键】收集概率用于绘图
            all_ensemble_probs.append(avg_probs.cpu().numpy())
            
            # --- 策略 A: 标准集成 (Argmax) ---
            _, final_preds = torch.max(avg_probs, 1)
            
            # --- 策略 B: 阈值修正 (Threshold) + 双重确认 ---
            preds_thresh_batch = final_preds.clone()
            
            # 倒序应用阈值
            for config in reversed(threshold_configs):
                target_idx = config['idx']
                thresh_val = config['thresh']
                target_name = config['name']
                
                target_probs = avg_probs[:, target_idx]
                
                # 【修改点】应用 NV < 0.8 的抑制逻辑
                if target_name == 'mel' and nv_idx != -1:
                    nv_probs = avg_probs[:, nv_idx]
                    # 只有当 (Mel > 0.25) 且 (NV < 0.8) 时，才认为是 Mel
                    mask = (target_probs > thresh_val) & (nv_probs < NV_SUPPRESSION_THRESHOLD)
                else:
                    # 其他类别 (bcc, akiec) 保持原样
                    mask = target_probs > thresh_val
                
                preds_thresh_batch[mask] = target_idx
            
            y_true.extend(labels.cpu().numpy())
            y_pred_ensemble.extend(final_preds.cpu().numpy())
            y_pred_thresh.extend(preds_thresh_batch.cpu().numpy())

    # =======================================================
    # 报告生成部分
    # =======================================================
    print("\n" + "="*60)
    print("[报告1] 架构性能对比 (单模型 vs 集成模型)")
    print("="*60)
    
    single_f1_scores = []
    for i, preds in enumerate(single_model_preds):
        f1 = f1_score(y_true, preds, average='macro')
        single_f1_scores.append(f1)
        
    ensemble_f1 = f1_score(y_true, y_pred_ensemble, average='macro')
    ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
    best_single_f1 = max(single_f1_scores)
    
    print(f"单模型最优F1:   {best_single_f1:.4f}")
    print(f"多模型集成F1:      {ensemble_f1:.4f}")
    print(f"性能提升:        {ensemble_f1 - best_single_f1:+.4f}")
    #print(f"Ensemble Accuracy:      {ensemble_acc:.4f}")

    # --- 策略对比 ---
    print("\n" + "="*60)
    print("[报告2] 临床策略对比 (默认策略 vs 风险阈值)")
    print("="*60)
    
    acc_thresh = accuracy_score(y_true, y_pred_thresh)
    f1_thresh = f1_score(y_true, y_pred_thresh, average='macro')
    
    print(f"{'性能指标':<16} | {'默认':<10} | {'阈值':<10} | {'变换'}")
    print("-" * 60)
    print(f"{'Accuracy':<20} | {ensemble_acc:.4f}{' '*4} | {acc_thresh:.4f}{' '*5} | {acc_thresh - ensemble_acc:+.4f}")
    print(f"{'Macro F1':<20} | {ensemble_f1:.4f}{' '*4} | {f1_thresh:.4f}{' '*5} | {f1_thresh - ensemble_f1:+.4f}")
    
    print("-" * 60)
    print("高危类别召回率变化：")
    print(f"{'类别':<8} | {'默认':<8} | {'阈值':<8} | {'变化'}")
    
    metrics_std = classification_report(y_true, y_pred_ensemble, target_names=class_names, output_dict=True)
    metrics_thr = classification_report(y_true, y_pred_thresh, target_names=class_names, output_dict=True)
    
    focus_classes = [c[0] for c in PRIORITY_THRESHOLDS]
    for cls in focus_classes:
        if cls in class_names:
            r_std = metrics_std[cls]['recall']
            r_thr = metrics_thr[cls]['recall']
            print(f"{cls:<10} | {r_std:.4f}{' '*4} | {r_thr:.4f}{' '*5} | {r_thr - r_std:+.4f}")

    print("\n" + "="*60)
    print("[报告3] 详细分类报告对比")
    print("="*60)
    print("\n[A] 标准集成策略")
    print(classification_report(y_true, y_pred_ensemble, target_names=class_names, digits=4))
    print("\n[B] 优先级阈值策略")
    print(classification_report(y_true, y_pred_thresh, target_names=class_names, digits=4))
    print("="*60)

    # --- 自动绘图 ---
    print("\n🎨 正在绘制毕设插图 (ROC/混淆矩阵/阈值分析)...")
    try:
        all_ensemble_probs = np.concatenate(all_ensemble_probs, axis=0)
        plot_thesis_figures(y_true, all_ensemble_probs, single_f1_scores, class_names)
    except Exception as e:
        print(f"❌ 绘图出错: {e}")

if __name__ == '__main__':
    ensemble_inference()