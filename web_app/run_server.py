# -*- coding: utf-8 -*-
"""启动Web服务器的脚本"""
import os
import sys

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 切换到web_app目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 导入并运行app
import app

if __name__ == '__main__':
    print("Starting server...")
    app.app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
