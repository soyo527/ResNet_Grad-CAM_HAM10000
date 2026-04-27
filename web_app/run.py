# -*- coding: utf-8 -*-
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Change to web_app directory
os.chdir(r'D:\软件\工作文件夹\本科毕设\my_project\web_app')
sys.path.insert(0, r'D:\软件\工作文件夹\本科毕设\my_project\web_app')

# Run the app
exec(open('app.py', encoding='utf-8').read())
