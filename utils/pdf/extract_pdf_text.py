#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取PDF文件中的文本内容，用于分析专家的分析思路
"""

import os
from PyPDF2 import PdfReader
import json

def extract_pdf_text(pdf_path):
    """
    提取单个PDF文件的文本内容
    
    参数:
        pdf_path: PDF文件路径
        
    返回:
        str: 提取的文本内容
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n=== 第 {page_num + 1} 页 ===\n"
                text += page_text
        return text
    except Exception as e:
        print(f"提取PDF文件 {pdf_path} 失败: {e}")
        return ""

def extract_all_pdfs(pdf_dir, output_file):
    """
    提取指定目录下所有PDF文件的文本内容
    
    参数:
        pdf_dir: PDF文件目录
        output_file: 输出JSON文件路径
        
    返回:
        dict: 包含所有PDF文本内容的字典
    """
    pdf_texts = {}
    
    # 获取目录下所有PDF文件
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"正在提取 {pdf_file}...")
        text = extract_pdf_text(pdf_path)
        pdf_texts[pdf_file] = text
    
    # 保存提取的文本内容到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pdf_texts, f, ensure_ascii=False, indent=4)
    
    print(f"所有PDF文件的文本内容已保存到 {output_file}")
    return pdf_texts

if __name__ == "__main__":
    # 设置PDF目录和输出文件路径
    PDF_DIR = "/Users/Williamhiler/Documents/my-project/train/pdf"
    OUTPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    
    # 提取所有PDF文件的文本内容
    extract_all_pdfs(PDF_DIR, OUTPUT_FILE)