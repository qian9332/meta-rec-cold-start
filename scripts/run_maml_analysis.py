#!/usr/bin/env python3
"""MAML vs FOMAML 对比实验脚本"""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.run_full_pipeline import run_experiment

if __name__ == "__main__":
    run_experiment()
