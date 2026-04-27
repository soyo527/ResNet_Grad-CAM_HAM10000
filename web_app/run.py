# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

WEB_APP_DIR = Path(__file__).resolve().parent
os.chdir(WEB_APP_DIR)
sys.path.insert(0, str(WEB_APP_DIR))

# Run the app
exec(open('app.py', encoding='utf-8').read())
