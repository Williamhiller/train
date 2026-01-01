#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步训练脚本 - Colab版本 - 使用专家数据进行初始训练

功能：
1. 加载指令-回答格式的专家数据
2. 配置LoRA参数
3. 对Qwen模型进行初始微调
4. 保存LoRA权重

使用方法：在Colab中运行
python train_lora_colab.py
"""

import torch
import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ==================== Colab配置参数 ====================
class Config:
    # 模型配置 - Colab版本使用Hugging Face模型
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"  # 从Hugging Face下载
    max_seq_length = 2048
    
    # 数据配置 - Colab版本
    data_path = "./v5/data/expert_data/qwen_finetune_data.json"  # 专家数据路径（Colab本地路径）
    
    # LoRA配置
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
    output_dir = "./out/qwen_lora_final"
    logging_steps = 10

# ==================== 辅助函数 ====================

def load_data(file_path):
    """加载训练数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_data(data):
    """处理数据为训练格式"""
    processed_data = []
    for item in data:
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]
        
        # 格式化样本
        text = f"### 指令：\n{instruction}\n\n### 输入：\n{input_text}\n\n### 回答：\n{output_text}"
        processed_data.append({"text": text})
    return processed_data

# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("第一步训练：使用专家数据进行初始微调（Colab版本）")
    print("=" * 60)
    
    # 1. 检测设备
    print(f"\n1. 检测训练设备")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"   - 设备: {device}")
    print(f"   - 数据类型: {dtype}")
    
    # 2. 加载数据
    print(f"\n2. 加载训练数据：{Config.data_path}")
    data = load_data(Config.data_path)
    print(f"   - 原始数据量: {len(data)} 条")
    
    # 3. 处理数据
    print(f"\n3. 处理训练数据")
    processed_data = process_data(data)
    dataset = Dataset.from_list(processed_data)
    print(f"   - 处理后数据量: {len(dataset)} 条")
    
    # 4. 加载模型和分词器（Colab版本：直接从Hugging Face下载）
    print(f"\n4. 从Hugging Face加载基础模型：{Config.base_model}")
    
    # 直接从Hugging Face下载模型，不需要本地路径
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # 5. 配置LoRA
    print(f"\n5. 配置LoRA参数")
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
    
    # 6. 配置训练器（适配Colab GPU）
    print(f"\n6. 配置训练器")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda example: example["text"],
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=Config.per_device_train_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            warmup_steps=Config.warmup_steps,
            max_steps=Config.max_steps,
            learning_rate=Config.learning_rate,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
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
        dataset_text_field="text",
        max_seq_length=Config.max_seq_length,
    )
    
    # 7. 开始训练
    print(f"\n7. 开始训练")
    print(f"   - 训练设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else device}")
    print(f"   - 训练批次大小: {Config.per_device_train_batch_size}")
    print(f"   - 梯度累积步数: {Config.gradient_accumulation_steps}")
    print(f"   - 最大训练步数: {Config.max_steps}")
    print("\n" + "=" * 60)
    
    trainer.train()
    
    # 8. 保存最终权重
    print(f"\n8. 保存LoRA权重到: {Config.output_dir}")
    model.save_pretrained(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)
    
    print(f"\n" + "=" * 60)
    print("第一步训练完成！")
    print(f"LoRA权重已保存到: {Config.output_dir}")
    print(f"可以在Colab中下载该目录，用于第二步训练")
    print("=" * 60)


if __name__ == "__main__":
    main()