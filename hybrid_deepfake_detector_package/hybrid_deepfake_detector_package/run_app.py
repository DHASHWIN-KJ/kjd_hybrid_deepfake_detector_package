#!/usr/bin/env python
import os
import sys

# Ensure package root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.gradio_app import demo, theme

if __name__ == '__main__':
    print('\n' + '=' * 70)
    print('🚀 Launching Hybrid Multi-Scale Deepfake Forensic Web App...')
    print('Open your browser at: http://127.0.0.1:7860')
    print('=' * 70 + '\n')
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True, theme=theme)
