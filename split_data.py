import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = BASE_DIR / "odata"
METADATA_PATH = ORIGINAL_DATA_DIR / "HAM10000_metadata.csv"
SOURCE_DIRS = [
    ORIGINAL_DATA_DIR / "HAM10000_images_part_1",
    ORIGINAL_DATA_DIR / "HAM10000_images_part_2"
]
OUTPUT_DIR = BASE_DIR / "data"
SEED = 42
# ===========================================

def main():
    print(f"HAM10000 数据集清洗与严格划分脚本")
    
    # 1. 建立索引 & 缺失检查
    print("1. 索引原始图片...")
    image_path_map = {}
    for src_dir in SOURCE_DIRS:
        if src_dir.exists():
            for p in src_dir.glob("*.jpg"):
                image_path_map[p.stem] = p
    
    # 2. 读取 CSV & 关联路径
    df = pd.read_csv(METADATA_PATH)
    total_metadata = len(df)
    
    df['path'] = df['image_id'].map(image_path_map)
    
    # --- 💡 恢复功能：缺失文件报警 ---
    missing_df = df[df['path'].isna()]
    if not missing_df.empty:
        print(f"警告: 发现 {len(missing_df)} 条元数据缺失对应图片文件！")
        # df = df.dropna(subset=['path']) # 如果你想强制过滤，取消注释
        # 但通常建议先检查下载是否完整
    else:
        print("文件完整性检查通过，无缺失。")
        
    df = df.dropna(subset=['path']) # 确保后续不报错

    # 3. 按 Lesion ID 划分 (核心防泄露逻辑)
    print("2. 执行 Lesion-ID 级分层抽样...")
    lesion_df = df.drop_duplicates(subset=['lesion_id'])
    
    # 8:1:1 划分
    train_ids, temp_ids = train_test_split(lesion_df['lesion_id'], test_size=0.2, random_state=SEED, stratify=lesion_df['dx'])
    temp_df = lesion_df[lesion_df['lesion_id'].isin(temp_ids)]
    val_ids, test_ids = train_test_split(temp_df['lesion_id'], test_size=0.5, random_state=SEED, stratify=temp_df['dx'])
    
    split_map = {
        'train': set(train_ids),
        'val': set(val_ids),
        'test': set(test_ids)
    }

    # 4. 执行复制
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    print("3. 物理复制文件...")
    # 用于统计每个集合中，每个类别的数量
    stats = {
        'train': {}, 'val': {}, 'test': {}
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lid = row['lesion_id']
        dx = row['dx']
        
        if lid in split_map['train']: split = 'train'
        elif lid in split_map['val']: split = 'val'
        elif lid in split_map['test']: split = 'test'
        else: continue
            
        # 记录统计
        stats[split][dx] = stats[split].get(dx, 0) + 1
        
        # 复制
        dest = OUTPUT_DIR / split / dx
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row['path'], dest / (row['image_id'] + ".jpg"))

    # --- 💡 恢复功能：详细的分病种统计表格 ---
    print("\n[最终数据分布报告]")
    print(f"{'Class':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 50)
    
    all_classes = sorted(df['dx'].unique())
    total_train = total_val = total_test = 0
    
    for cls in all_classes:
        n_train = stats['train'].get(cls, 0)
        n_val = stats['val'].get(cls, 0)
        n_test = stats['test'].get(cls, 0)
        n_total = n_train + n_val + n_test
        
        total_train += n_train
        total_val += n_val
        total_test += n_test
        
        print(f"{cls:<10} | {n_train:<8} | {n_val:<8} | {n_test:<8} | {n_total:<8}")
        
    print("-" * 50)
    print(f"{'ALL':<10} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {total_train+total_val+total_test:<8}")
    print(f"\n数据集已生成至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()