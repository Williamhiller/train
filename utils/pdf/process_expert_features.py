#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合的专家特征处理脚本，包含从PDF提取文本、清洗文本到提取特征的完整流程
"""

import os
import sys
import json
import argparse

# 确保当前目录在路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_pdf_text import extract_all_pdfs
from clean_pdf_text import PDFTextCleaner
from extract_expert_features import ExpertFeatureExtractor

class ExpertFeatureProcessor:
    """
    专家特征处理完整流程类
    """
    
    def __init__(self, pdf_dir: str, output_dir: str):
        """
        初始化处理器
        
        参数:
            pdf_dir: PDF文件目录
            output_dir: 输出文件目录
        """
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件路径设置
        self.raw_texts_file = os.path.join(output_dir, 'pdf_texts.json')
        self.cleaned_texts_file = os.path.join(output_dir, 'cleaned_pdf_texts.json')
        self.expert_features_file = os.path.join(output_dir, 'expert_features_analysis.json')
    
    def run_full_process(self, skip_text_extraction: bool = False, skip_text_cleaning: bool = False) -> None:
        """
        运行完整的处理流程
        
        参数:
            skip_text_extraction: 是否跳过文本提取（如果已经有提取好的文本）
            skip_text_cleaning: 是否跳过文本清洗（如果已经有清洗好的文本）
        """
        print("=== 专家特征处理流程开始 ===")
        
        # 步骤1: 提取PDF文本
        if not skip_text_extraction:
            self._extract_pdf_texts()
        else:
            print("跳过PDF文本提取步骤")
        
        # 步骤2: 清洗文本
        if not skip_text_cleaning:
            self._clean_texts()
        else:
            print("跳作文本清洗步骤")
        
        # 步骤3: 提取专家特征
        self._extract_expert_features()
        
        print("\n=== 专家特征处理流程完成 ===")
    
    def _extract_pdf_texts(self) -> None:
        """
        从PDF文件中提取文本
        """
        print("\n步骤1: 提取PDF文本")
        print(f"正在从 {self.pdf_dir} 目录提取PDF文本...")
        
        # 调用现有的文本提取函数
        extract_all_pdfs(self.pdf_dir, self.raw_texts_file)
        
        print(f"PDF文本已提取到 {self.raw_texts_file}")
    
    def _clean_texts(self) -> None:
        """
        清洗提取的文本
        """
        print("\n步骤2: 清洗文本")
        print(f"正在清洗 {self.raw_texts_file} 中的文本...")
        
        # 加载原始文本
        with open(self.raw_texts_file, 'r', encoding='utf-8') as f:
            pdf_texts = json.load(f)
        
        # 创建清洗器实例
        cleaner = PDFTextCleaner()
        
        # 清洗文本
        cleaned_texts = cleaner.clean_pdf_texts(pdf_texts)
        
        # 保存清洗后的文本
        with open(self.cleaned_texts_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_texts, f, ensure_ascii=False, indent=4)
        
        print(f"清洗后的文本已保存到 {self.cleaned_texts_file}")
    
    def _extract_expert_features(self) -> None:
        """
        从清洗后的文本中提取专家特征
        """
        print("\n步骤3: 提取专家特征")
        
        # 创建提取器实例
        extractor = ExpertFeatureExtractor(use_cleaned_text=True)
        
        # 运行特征提取
        extractor.run_analysis(self.cleaned_texts_file, self.expert_features_file)
        
        print(f"专家特征已提取到 {self.expert_features_file}")
    
    def verify_results(self) -> None:
        """
        验证处理结果
        """
        print("\n=== 结果验证 ===")
        
        # 检查文件是否存在
        files_to_check = [
            ("原始PDF文本", self.raw_texts_file),
            ("清洗后的文本", self.cleaned_texts_file),
            ("专家特征分析", self.expert_features_file)
        ]
        
        for file_desc, file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"✓ {file_desc} 文件存在: {file_path} ({file_size:.2f} KB)")
            else:
                print(f"✗ {file_desc} 文件不存在: {file_path}")
        
        # 检查专家特征文件内容
        if os.path.exists(self.expert_features_file):
            with open(self.expert_features_file, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
            
            print("\n专家特征分析内容摘要:")
            
            # 赔率特征
            odds_count = len(features_data['expert_features']['odds_features']['values'])
            print(f"- 赔率相关特征: {odds_count}个")
            
            # 球队表现特征
            team_count = len(features_data['expert_features']['team_performance_features']['values'])
            print(f"- 球队表现相关特征: {team_count}个")
            
            # 历史对阵特征
            head_count = len(features_data['expert_features']['head_to_head_features']['values'])
            print(f"- 历史对阵相关特征: {head_count}个")
            
            # 分析规则
            rules_count = len(features_data['expert_features']['analysis_rules']['complete_rules'])
            print(f"- 完整分析规则: {rules_count}个")

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description='专家特征处理脚本')
    parser.add_argument('--pdf-dir', type=str, default='/Users/Williamhiler/Documents/my-project/train/pdf',
                      help='PDF文件目录路径')
    parser.add_argument('--output-dir', type=str, default='/Users/Williamhiler/Documents/my-project/train/train-data/expert',
                      help='输出文件目录路径')
    parser.add_argument('--skip-text-extraction', action='store_true',
                      help='跳过PDF文本提取步骤')
    parser.add_argument('--skip-text-cleaning', action='store_true',
                      help='跳作文本清洗步骤')
    parser.add_argument('--verify-only', action='store_true',
                      help='仅验证结果，不运行处理流程')
    
    args = parser.parse_args()
    
    # 创建处理器实例
    processor = ExpertFeatureProcessor(args.pdf_dir, args.output_dir)
    
    if args.verify_only:
        # 仅验证结果
        processor.verify_results()
    else:
        # 运行完整流程
        processor.run_full_process(args.skip_text_extraction, args.skip_text_cleaning)
        
        # 验证结果
        processor.verify_results()

if __name__ == "__main__":
    main()
