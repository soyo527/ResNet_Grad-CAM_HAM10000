# -*- coding: utf-8 -*-
"""测试应用是否可以正常启动"""
import sys
import os
from pathlib import Path

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 切换到 web_app 目录
WEB_APP_DIR = Path(__file__).resolve().parent
os.chdir(WEB_APP_DIR)
sys.path.insert(0, str(WEB_APP_DIR))

try:
    # 导入app模块
    import app
    
    print("App module imported successfully")
    print(f"Model directory: {app.MODEL_DIR}")
    print(f"Number of classes: {app.NUM_CLASSES}")
    print(f"Device: {app.DEVICE}")
    
    # 测试模型加载
    print("\nTesting model loading...")
    app.load_models()
    
    print("\nAll tests passed! The application should work correctly.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
