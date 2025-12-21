#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗PDF提取的文本内容，去除噪声并提高特征提取的质量
"""

import re
import json
import string
from typing import Dict, List, Optional

class PDFTextCleaner:
    """
    PDF文本清洗类，用于处理从PDF提取的原始文本
    """
    
    def __init__(self):
        # 定义需要移除的噪声模式
        self.noise_patterns = [
            r'=== 第 \d+ 页 ===',  # 页码标记
            r'\s+',  # 多余空白字符（多个空格、制表符、换行符等）
            r'\n+',  # 多余换行符
            r'\r+',  # 回车符
        ]
        
        # 定义需要标准化的文本模式
        self.standardization_patterns = [
            (r'近(\d+)場', r'近\1场'),  # 繁体转简体
            (r'勝率', r'胜率'),
            (r'積分', r'积分'),
            (r'排名', r'排名'),  # 已为简体，保持不变
            (r'進球', r'进球'),
            (r'失球', r'失球'),
            (r'主場', r'主场'),
            (r'客場', r'客场'),
            (r'歷史交鋒', r'历史交锋'),
            (r'交鋒記錄', r'交锋记录'),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        清洗单篇PDF文本
        
        参数:
            text: 原始PDF文本
            
        返回:
            str: 清洗后的文本
        """
        cleaned_text = text
        
        # 移除噪声
        for pattern in self.noise_patterns:
            cleaned_text = re.sub(pattern, ' ', cleaned_text)
        
        # 文本标准化
        for pattern, replacement in self.standardization_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 移除多余标点符号
        cleaned_text = self._remove_excess_punctuation(cleaned_text)
        
        # 去除首尾空白
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _remove_excess_punctuation(self, text: str) -> str:
        """
        移除多余的标点符号
        
        参数:
            text: 待处理文本
            
        返回:
            str: 处理后的文本
        """
        # 移除连续的标点符号
        text = re.sub(r'([{}])\1+'.format(re.escape(string.punctuation)), r'\1', text)
        # 移除句子开头和结尾的标点符号
        text = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', text)
        return text
    
    def clean_pdf_texts(self, pdf_texts: Dict[str, str]) -> Dict[str, str]:
        """
        清洗所有PDF文本
        
        参数:
            pdf_texts: 包含所有PDF文本的字典 {pdf文件名: 原始文本}
            
        返回:
            Dict[str, str]: 清洗后的PDF文本字典
        """
        cleaned_pdfs = {}
        
        for pdf_file, text in pdf_texts.items():
            cleaned_pdfs[pdf_file] = self.clean_text(text)
        
        return cleaned_pdfs
    
    def save_cleaned_texts(self, cleaned_texts: Dict[str, str], output_file: str) -> None:
        """
        保存清洗后的文本到JSON文件
        
        参数:
            cleaned_texts: 清洗后的PDF文本字典
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_texts, f, ensure_ascii=False, indent=4)
        print(f"清洗后的文本已保存到 {output_file}")

def load_pdf_texts(input_file: str) -> Dict[str, str]:
    """
    加载PDF文本文件
    
    参数:
        input_file: 输入文件路径
        
    返回:
        Dict[str, str]: PDF文本字典
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """
    主函数，用于演示文本清洗功能
    """
    # 设置输入输出文件路径
    INPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    OUTPUT_FILE = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/cleaned_pdf_texts.json"
    
    # 加载原始文本
    print("正在加载原始PDF文本...")
    pdf_texts = load_pdf_texts(INPUT_FILE)
    
    # 创建清洗器实例
    cleaner = PDFTextCleaner()
    
    # 清洗文本
    print("正在清洗PDF文本...")
    cleaned_texts = cleaner.clean_pdf_texts(pdf_texts)
    
    # 保存清洗后的文本
    cleaner.save_cleaned_texts(cleaned_texts, OUTPUT_FILE)
    
    # 打印清洗前后的对比
    print("\n=== 清洗前后对比示例 ===")
    for pdf_file in list(pdf_texts.keys())[:1]:  # 只显示第一个文件的对比
        original = pdf_texts[pdf_file][:200]  # 只显示前200个字符
        cleaned = cleaned_texts[pdf_file][:200]
        print(f"\n原始文本前200字符:")
        print(original)
        print(f"\n清洗后文本前200字符:")
        print(cleaned)

if __name__ == "__main__":
    main()
