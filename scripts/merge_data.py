import os
import shutil
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
source_root = str(PROJECT_ROOT / 'data')          # 你现在的train/val所在的根目录
target_dir = str(PROJECT_ROOT / 'data' / 'all_data')  # 你想把数据合并到的新位置
# ===========================================

def merge_datasets():
    if not os.path.exists(source_root):
        print(f"错误：找不到源目录 {source_root}")
        return

    # 1. 自动检测类别名称 (基于train文件夹)
    train_dir = os.path.join(source_root, 'train')
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"检测到类别: {classes}")

    # 2. 创建目标目录结构
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        for cls in classes:
            os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

    # 3. 开始合并
    # 遍历 train 和 val
    for split in ['train', 'val']:
        split_path = os.path.join(source_root, split)
        if not os.path.exists(split_path):
            print(f"警告：找不到 {split} 文件夹，跳过")
            continue

        print(f"正在处理 {split} 数据...")
        
        for cls in classes:
            cls_source_path = os.path.join(split_path, cls)
            cls_target_path = os.path.join(target_dir, cls)
            
            if not os.path.exists(cls_source_path):
                continue

            files = os.listdir(cls_source_path)
            for fname in tqdm(files, desc=f"{split}/{cls}"):
                src_file = os.path.join(cls_source_path, fname)
                dst_file = os.path.join(cls_target_path, fname)
                
                # 如果是文件才复制
                if os.path.isfile(src_file):
                    # 防止文件名冲突（虽然HAM10000ID唯一，但在不同源可能重名）
                    if os.path.exists(dst_file):
                        # 如果冲突，在文件名前加 split 前缀
                        fname_new = f"{split}_{fname}"
                        dst_file = os.path.join(cls_target_path, fname_new)
                    
                    shutil.copy2(src_file, dst_file) # copy2 保留文件元数据

    print("\n合并完成！")
    print(f"新数据集位置: {target_dir}")
    # 打印统计信息
    total_imgs = 0
    for cls in classes:
        count = len(os.listdir(os.path.join(target_dir, cls)))
        print(f"  - {cls}: {count} 张")
        total_imgs += count
    print(f"总计图片: {total_imgs} 张")

if __name__ == '__main__':
    merge_datasets()
