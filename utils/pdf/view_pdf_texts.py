#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看提取的PDF文本内容，了解专家的分析思路
"""

import json

def view_pdf_texts(json_path, max_chars=2000):
    """
    查看提取的PDF文本内容
    
    参数:
        json_path: JSON文件路径
        max_chars: 每个PDF文件显示的最大字符数
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        pdf_texts = json.load(f)
    
    for pdf_file, text in pdf_texts.items():
        print(f"\n=== {pdf_file} ===")
        print(f"总字符数: {len(text)}")
        print("\n前 {max_chars} 字符:")
        print(text[:max_chars])
        print("\n" + "="*50)

if __name__ == "__main__":
    JSON_PATH = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    view_pdf_texts(JSON_PATH, max_chars=1000)
