#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专家数据预处理脚本：将PDF文件转换为Qwen可以学习的格式
"""

import os
import sys
import yaml
import json
import glob
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed, installing...")
    os.system("pip install PyPDF2")
    import PyPDF2


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF文件中提取文本"""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """将文本分块，用于训练"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def create_training_data(pdf_texts: Dict[str, str], output_path: str) -> List[Dict]:
    """创建训练数据
    
    Args:
        pdf_texts: PDF文件路径->文本的字典
        output_path: 输出路径
        
    Returns:
        训练数据列表
    """
    training_data = []
    
    for pdf_path, text in tqdm(pdf_texts.items(), desc="Processing PDF files"):
        if not text.strip():
            print(f"Warning: Empty text in {pdf_path}, skipping...")
            continue
        
        # 将文本分块
        chunks = chunk_text(text, chunk_size=2000, overlap=200)
        
        # 创建训练样本（prompt-response对）
        for i, chunk in enumerate(chunks):
            # 创建prompt（问题）
            prompt = f"请基于以下专家分析，提取足球比赛预测的关键规则和模式：{chunk}"
            
            # 创建response（答案）
            response = chunk
            
            training_data.append({
                "pdf_file": pdf_path,
                "chunk_id": i,
                "prompt": prompt,
                "response": response
            })
    
    # 保存训练数据
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "expert_training_data.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练数据已保存到: {output_file}")
    print(f"总计 {len(training_data)} 个训练样本")
    
    return training_data


def preprocess_expert_data(config: Dict):
    """预处理专家数据"""
    print("=" * 80)
    print("专家数据预处理")
    print("=" * 80)
    
    pdf_directory = config["data"]["expert_data_path"]
    output_path = config["data"].get("expert_processed_path", "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_data")
    
    print(f"PDF目录: {pdf_directory}")
    print(f"输出路径: {output_path}")
    
    # 查找所有PDF文件
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    if len(pdf_files) == 0:
        print("警告: 没有找到PDF文件")
        return {}
    
    # 提取所有PDF文件的文本
    pdf_texts = {}
    for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
        text = extract_text_from_pdf(pdf_file)
        pdf_texts[pdf_file] = text
    
    # 创建训练数据
    training_data = create_training_data(pdf_texts, output_path)
    
    print(f"预处理完成！")
    print(f"总PDF文件数: {len(pdf_texts)}")
    print(f"总训练样本数: {len(training_data)}")
    
    return {
        "pdf_files": pdf_files,
        "pdf_texts": pdf_texts,
        "training_data": training_data
    }


def main():
    """主函数"""
    config_path = "configs/v5_config.yaml"
    
    print("开始专家数据预处理...")
    print(f"配置文件: {config_path}")
    
    config = load_config(config_path)
    
    try:
        result = preprocess_expert_data(config)
        
        print("\n" + "=" * 80)
        print("专家数据预处理完成!")
        print("=" * 80)
        print(f"训练数据已保存，可以开始训练Qwen模型了")
        
    except Exception as e:
        print(f"\n专家数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
