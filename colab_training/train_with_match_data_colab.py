#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二步训练脚本 - Colab版本 - 使用比赛数据进行进一步训练

功能：
1. 读取英超各赛季的聚合数据
2. 提取赔率、球队状态等特征
3. 构建适合Qwen模型训练的数据集
4. 基于第一步训练好的LoRA权重进行进一步微调

使用方法：在Colab中运行
python train_with_match_data_colab.py
"""

import torch
import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# ==================== Colab配置参数 ====================
class Config:
    # 模型配置 - Colab版本使用Hugging Face模型
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"  # 从Hugging Face下载
    lora_weights = "./out/qwen_lora_final"  # 第一步训练的LoRA权重（Colab本地路径）
    max_seq_length = 2048
    
    # 比赛数据配置 - Colab版本
    match_data_path = "./examples"  # 比赛数据目录（加载所有赛季数据）
    
    # LoRA配置（进一步微调）
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    # 训练参数 - 适配Colab GPU
    per_device_train_batch_size = 8  # 增大批次大小，利用GPU内存
    gradient_accumulation_steps = 2  # 减少梯度累积步数
    warmup_steps = 50
    max_steps = 300
    learning_rate = 1e-5
    weight_decay = 0.01
    
    # 输出配置 - Colab本地路径
    output_dir = "./out/match_finetune"
    logging_steps = 10

# ==================== 辅助函数 ====================

def load_match_data(file_path_or_dir):
    """加载比赛数据
    
    支持两种模式：
    1. 加载单个文件
    2. 加载目录下所有_aggregated.json结尾的文件并合并
    
    Args:
        file_path_or_dir: 文件路径或目录路径
        
    Returns:
        dict: 合并后的比赛数据
    """
    # 检查是文件还是目录
    if os.path.isfile(file_path_or_dir):
        # 加载单个文件
        with open(file_path_or_dir, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.isdir(file_path_or_dir):
        # 加载目录下所有符合条件的文件
        merged_data = {}
        file_count = 0
        
        # 遍历目录下所有文件
        for filename in os.listdir(file_path_or_dir):
            if filename.endswith('_aggregated.json'):
                file_path = os.path.join(file_path_or_dir, filename)
                print(f"   - 加载文件：{filename}")
                
                # 加载单个文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 合并数据
                merged_data.update(data)
                file_count += 1
        
        print(f"   - 共加载 {file_count} 个文件")
        return merged_data
    else:
        raise ValueError(f"{file_path_or_dir} 不是有效的文件或目录路径")


def extract_features(match_id, match_info):
    """从比赛信息中提取特征"""
    # 基本信息
    match_time = match_info.get("matchTime", "")
    home_team = match_info.get("homeTeamId", "")
    away_team = match_info.get("awayTeamId", "")
    result = match_info.get("result", "")
    home_score = match_info.get("homeScore", 0)
    away_score = match_info.get("awayScore", 0)
    
    # 提取赔率信息
    odds_info = []
    details = match_info.get("details", {})
    odds = details.get("odds", {})
    for bookie_id, odds_list in odds.items():
        if odds_list and isinstance(odds_list, list):
            latest_odds = odds_list[-1]  # 获取最新赔率
            if len(latest_odds) >= 3:
                odds_info.append(f"庄家{bookie_id}：胜{latest_odds[0]}，平{latest_odds[1]}，负{latest_odds[2]}")
    
    # 提取球队历史数据
    history = details.get("history", {})
    home_data = history.get("homeData", [])
    away_data = history.get("awayData", [])
    
    # 构建特征文本
    features = f"比赛ID：{match_id}\n"
    features += f"比赛时间：{match_time}\n"
    features += f"对阵：{home_team} VS {away_team}\n"
    features += f"比赛结果：{home_score}-{away_score}（{['平局', '主胜', '客胜'][result] if result in [0,1,2] else '未知'}\n"
    features += f"赔率信息：{'; '.join(odds_info)}\n"
    features += f"主队近期战绩：{str(home_data[:3])}\n"
    features += f"客队近期战绩：{str(away_data[:3])}\n"
    
    return features


def build_training_dataset(match_data):
    """构建训练数据集"""
    training_samples = []
    
    for match_id, match_info in match_data.items():
        # 提取基本信息
        home_team = match_info.get("homeTeamId", "")
        away_team = match_info.get("awayTeamId", "")
        result = match_info.get("result", "")
        home_score = match_info.get("homeScore", 0)
        away_score = match_info.get("awayScore", 0)
        
        # 提取赔率信息
        odds_info = []
        details = match_info.get("details", {})
        odds = details.get("odds", {})
        for bookie_id, odds_list in odds.items():
            if odds_list and isinstance(odds_list, list):
                latest_odds = odds_list[-1]  # 获取最新赔率
                if len(latest_odds) >= 3:
                    odds_info.append(f"庄家{bookie_id}：胜{latest_odds[0]}，平{latest_odds[1]}，负{latest_odds[2]}")
        
        # 提取球队历史数据
        history = details.get("history", {})
        home_data = history.get("homeData", [])
        away_data = history.get("awayData", [])
        
        # 提取特征
        features = extract_features(match_id, match_info)
        
        # 构建指令和期望输出
        instruction = f"请基于以下比赛数据，分析这场比赛的赔率变化和球队状态，并预测比赛结果。\n\n{features}"
        
        # 构建回答
        answer = f"根据比赛数据和赔率分析，这场比赛的结果是{home_team} {home_score}-{away_score} {away_team}，最终结果为{['平局', '主胜', '客胜'][result] if result in [0,1,2] else '未知'}。\n\n赔率分析：{'; '.join(odds_info)}\n球队状态分析：主队近期战绩{str(home_data[:3])}，客队近期战绩{str(away_data[:3])}。"
        
        # 格式化样本为instruction-input-output格式
        sample = {
            "text": f"### 指令：\n{instruction}\n\n### 回答：\n{answer}"
        }
        training_samples.append(sample)
    
    return training_samples

# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("第二步训练：使用比赛数据进行微调（Colab版本）")
    print("=" * 60)
    
    # 1. 加载比赛数据
    print(f"\n1. 加载比赛数据：{Config.match_data_path}")
    match_data = load_match_data(Config.match_data_path)
    print(f"   - 加载比赛场次：{len(match_data)}")
    
    # 2. 构建训练数据集
    print(f"\n2. 构建训练数据集")
    training_samples = build_training_dataset(match_data)
    print(f"   - 生成训练样本：{len(training_samples)}")
    
    # 3. 转换为Dataset格式
    dataset = Dataset.from_list(training_samples)
    
    # 4. 检测设备
    print(f"\n3. 检测训练设备")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"   - 设备: {device}")
    print(f"   - 数据类型: {dtype}")
    
    # 5. 加载模型和分词器（Colab版本：直接从Hugging Face下载）
    print(f"\n4. 从Hugging Face加载基础模型：{Config.base_model}")
    
    # 直接从Hugging Face下载模型，不需要本地路径
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # 6. 加载第一步训练的LoRA权重
    print(f"\n5. 加载第一步LoRA权重：{Config.lora_weights}")
    if os.path.exists(Config.lora_weights):
        model = PeftModel.from_pretrained(
            model, 
            Config.lora_weights,
            torch_dtype=dtype,
        )
        print(f"   ✓ 成功加载第一步LoRA权重")
    else:
        print(f"   ⚠️  第一步LoRA权重不存在：{Config.lora_weights}")
        print(f"   ⚠️  将使用基础模型进行训练")
    
    # 7. 配置LoRA（进一步微调）
    print(f"\n6. 配置LoRA参数")
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 8. 配置训练器（适配Colab GPU）
    print(f"\n7. 配置训练器")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda example: example["text"],
        max_seq_length=Config.max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=Config.per_device_train_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            warmup_steps=Config.warmup_steps,
            max_steps=Config.max_steps,
            learning_rate=Config.learning_rate,
            fp16=torch.cuda.is_available(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=Config.logging_steps,
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay=Config.weight_decay,
            lr_scheduler_type="cosine",
            output_dir=Config.output_dir,
            report_to="none",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
        ),
    )
    
    # 9. 开始训练
    print(f"\n8. 开始训练")
    print(f"   - 训练设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else device}")
    print(f"   - 训练批次大小: {Config.per_device_train_batch_size}")
    print(f"   - 梯度累积步数: {Config.gradient_accumulation_steps}")
    print(f"   - 最大训练步数: {Config.max_steps}")
    print("\n" + "=" * 60)
    
    trainer.train()
    
    # 10. 保存最终权重
    print(f"\n9. 保存微调后的LoRA权重到: {Config.output_dir}")
    final_output_dir = os.path.join(Config.output_dir, "final_lora")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\n" + "=" * 60)
    print("第二步训练完成！")
    print(f"最终LoRA权重已保存到: {final_output_dir}")
    print(f"可以在Colab中下载该目录，用于后续部署或本地测试")
    print("=" * 60)


if __name__ == "__main__":
    main()