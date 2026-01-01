#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLM专家分析处理现有专家数据
生成结构化的专家特征用于v4.0.1模型训练
"""

import os
import json
import time
import argparse
from typing import Dict, List
from llm_expert_analysis import LLMExpertAnalyzer

def split_pdf_into_chapters(pdf_content: str, max_chapter_length: int = 1000) -> List[str]:
    """
    将PDF内容分割为适合LLM分析的章节
    
    参数:
        pdf_content: PDF文本内容
        max_chapter_length: 每个章节的最大长度
    
    返回:
        List[str]: 章节列表
    """
    chapters = []
    current_chapter = ""
    
    # 按段落分割
    paragraphs = pdf_content.split('\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果当前章节加上新段落超过最大长度，则保存当前章节
        if len(current_chapter) + len(para) + 1 > max_chapter_length:
            if current_chapter.strip():
                chapters.append(current_chapter.strip())
            current_chapter = para
        else:
            current_chapter += "\n" + para
    
    # 添加最后一个章节
    if current_chapter.strip():
        chapters.append(current_chapter.strip())
    
    return chapters

def generate_combined_analysis(output_dir: str):
    """
    生成综合分析报告
    
    参数:
        output_dir: 分析结果目录
    """
    combined_analysis = {
        "total_files": 0,
        "total_chapters": 0,
        "average_confidence": 0.0,
        "prediction_distribution": {},
        "sentiment_distribution": {},
        "key_factors_frequency": {},
        "files_analysis": {}
    }
    
    total_confidence = 0
    total_chapters = 0
    key_factors_counter = {}
    
    # 遍历所有分析结果文件
    for filename in os.listdir(output_dir):
        if not filename.endswith("_llm_analysis.json") or filename == "combined_llm_analysis.json":
            continue
        
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_results = json.load(f)
        
        combined_analysis["total_files"] += 1
        combined_analysis["total_chapters"] += len(file_results)
        total_chapters += len(file_results)
        
        # 统计文件级别的分析
        file_confidence = []
        file_predictions = {}
        file_sentiments = {}
        
        for result in file_results:
            analysis = result["analysis_result"]
            
            # 统计信心评分
            total_confidence += analysis["confidence_score"]
            file_confidence.append(analysis["confidence_score"])
            
            # 统计预测分布
            prediction = analysis["prediction"]
            combined_analysis["prediction_distribution"][prediction] = \
                combined_analysis["prediction_distribution"].get(prediction, 0) + 1
            file_predictions[prediction] = file_predictions.get(prediction, 0) + 1
            
            # 统计情感分布
            sentiment = analysis["sentiment"]
            combined_analysis["sentiment_distribution"][sentiment] = \
                combined_analysis["sentiment_distribution"].get(sentiment, 0) + 1
            file_sentiments[sentiment] = file_sentiments.get(sentiment, 0) + 1
            
            # 统计关键因素频率
            for factor in analysis["key_factors"]:
                key_factors_counter[factor] = key_factors_counter.get(factor, 0) + 1
        
        # 保存文件级别的分析
        combined_analysis["files_analysis"][filename] = {
            "total_chapters": len(file_results),
            "average_confidence": sum(file_confidence) / len(file_confidence) if file_confidence else 0,
            "prediction_distribution": file_predictions,
            "sentiment_distribution": file_sentiments
        }
    
    # 计算平均信心评分
    if total_chapters > 0:
        combined_analysis["average_confidence"] = total_confidence / total_chapters
    
    # 保存关键因素频率
    combined_analysis["key_factors_frequency"] = dict(sorted(
        key_factors_counter.items(), key=lambda x: x[1], reverse=True
    ))
    
    # 保存综合分析报告
    output_path = os.path.join(output_dir, "combined_llm_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 生成综合分析报告: {output_path}")

def process_expert_data(pdf_files=None):
    """
    处理专家数据，生成结构化的LLM分析结果
    
    参数:
        pdf_files: 要处理的PDF文件列表，如果为None则处理所有文件
    """
    print("=== 开始处理专家数据 ===")
    
    # 【配置】在这里指定要解析的文件名数组
    # 这个数组可以直接在代码中修改，也可以通过命令行参数覆盖
    CONFIGURED_FILES = [
        "欧赔核心思维",
        # 添加更多文件名或关键词，支持部分匹配
        # "欧赔招式版",
        # "六步解欧赔",
        # "基础版"
    ]
    
    # 初始化LLM专家分析器
    analyzer = LLMExpertAnalyzer()
    
    # 加载PDF文本数据
    pdf_texts_path = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
    with open(pdf_texts_path, 'r', encoding='utf-8') as f:
        pdf_texts = json.load(f)
    
    # 创建输出目录
    output_dir = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/llm_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定最终要处理的文件列表：命令行参数优先级高于配置文件
    final_pdf_files = None
    if pdf_files:
        # 使用命令行参数指定的文件
        final_pdf_files = pdf_files
        print(f"\n✓ 使用命令行指定的文件列表")
    elif CONFIGURED_FILES:
        # 使用配置文件中指定的文件
        final_pdf_files = CONFIGURED_FILES
        print(f"\n✓ 使用代码中配置的文件列表")
    
    # 过滤要处理的文件
    if final_pdf_files:
        # 只处理指定的文件
        filtered_pdf_texts = {}
        for pdf_name, pdf_content in pdf_texts.items():
            # 检查文件名是否在指定列表中（支持部分匹配）
            if any(file_name in pdf_name for file_name in final_pdf_files):
                filtered_pdf_texts[pdf_name] = pdf_content
        print(f"✓ 过滤后将处理 {len(filtered_pdf_texts)} 个文件")
    else:
        # 处理所有文件
        filtered_pdf_texts = pdf_texts
        print(f"\n✓ 处理所有文件 ({len(filtered_pdf_texts)} 个)")
    
    # 处理每个PDF文件
    for pdf_name, pdf_content in filtered_pdf_texts.items():
        print(f"\n--- 处理文件: {pdf_name} ---")
        
        # 分割PDF内容为章节
        chapters = split_pdf_into_chapters(pdf_content)
        total_chapters = len(chapters)
        print(f"共分割出 {total_chapters} 个章节")
        
        # 生成进度文件路径
        pdf_basename = os.path.splitext(pdf_name)[0]
        progress_file = os.path.join(output_dir, f"{pdf_basename}_progress.json")
        output_path = os.path.join(output_dir, f"{pdf_basename}_llm_analysis.json")
        
        # 检查是否有未完成的进度
        start_chapter = 0
        llm_analysis_results = []
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                start_chapter = progress_data.get('current_chapter', 0)
                llm_analysis_results = progress_data.get('results', [])
                
                if start_chapter > 0:
                    print(f"✓ 发现未完成进度，从章节 {start_chapter+1}/{total_chapters} 继续")
            except Exception as e:
                print(f"⚠ 读取进度文件失败: {e}，将从头开始")
                start_chapter = 0
                llm_analysis_results = []
        
        # 分析每个章节（从上次中断的地方继续）
        for i in range(start_chapter, total_chapters):
            chapter = chapters[i]
            if not chapter.strip():
                continue
            
            print(f"分析章节 {i+1}/{total_chapters}... ({(i+1)/total_chapters*100:.1f}% 完成)")
            
            # 使用LLM分析章节内容
            result = analyzer.analyze_expert_text(chapter)
            
            # 保存分析结果
            analysis_data = {
                "chapter_index": i + 1,
                "chapter_content": chapter[:200] + "..." if len(chapter) > 200 else chapter,
                "analysis_result": {
                    "confidence_score": result.confidence_score,
                    "prediction": result.prediction,
                    "reasoning_quality": result.reasoning_quality,
                    "key_factors": result.key_factors,
                    "risk_assessment": result.risk_assessment,
                    "sentiment": result.sentiment,
                    "timestamp": result.timestamp
                }
            }
            
            llm_analysis_results.append(analysis_data)
            
            # 实时保存进度
            progress_data = {
                'current_chapter': i + 1,
                'total_chapters': total_chapters,
                'results': llm_analysis_results,
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            # 避免过于频繁的请求
            time.sleep(1)
        
        # 保存最终分析结果到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 保存分析结果到: {output_path}")
        
        # 清理进度文件（任务完成）
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"✓ 清理进度文件: {progress_file}")
    
    # 生成综合分析报告
    generate_combined_analysis(output_dir)
    
    print("\n=== 专家数据处理完成 ===")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用LLM专家分析处理现有专家数据")
    parser.add_argument("--files", nargs="*", help="要处理的PDF文件名（支持部分匹配），不指定则处理所有文件")
    
    args = parser.parse_args()
    
    # 调用处理函数
    process_expert_data(args.files)