#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二步训练脚本 - 使用预处理后的比赛数据进行进一步训练

功能：
1. 读取预处理后的比赛数据（包含优化的特征提取和格式）
2. 基于第一步训练好的LoRA权重进行进一步微调
3. 支持GPU/CPU训练
"""

import torch
import json
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# ==================== 配置参数 ====================
class Config:
    # 模型配置
    base_model = "/Users/Williamhiler/Documents/my-project/train/models/cache/Qwen2.5-0.5B-Instruct"  # 绝对路径
    lora_weights = "/Users/Williamhiler/Documents/my-project/train/colab_training/out/qwen_lora_final"  # 绝对路径
    max_seq_length = 2048
    
    # 比赛数据配置
    match_data_path = "/Users/Williamhiler/Documents/my-project/train/colab_training/match/match_train_data.json"  # 绝对路径，指向预处理后的数据
    
    # LoRA配置（进一步微调）
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    # 训练参数
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    warmup_steps = 50
    max_steps = 300
    learning_rate = 1e-5
    weight_decay = 0.01
    
    # 输出配置
    output_dir = "/Users/Williamhiler/Documents/my-project/train/colab_training/out/match_finetune"  # 绝对路径
    logging_steps = 10

def load_processed_data(file_path):
    """加载预处理后的训练数据
    
    Args:
        file_path: 预处理后的数据文件路径
        
    Returns:
        list: 训练样本列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("第二步训练：使用比赛数据进行微调")
    print("=" * 60)
    
    # 1. 加载预处理后的比赛数据
    print(f"\n1. 加载预处理后的数据：{Config.match_data_path}")
    training_samples = load_processed_data(Config.match_data_path)
    print(f"   - 加载训练样本数量：{len(training_samples)}")
    
    # 2. 转换为Dataset格式
    dataset = Dataset.from_list(training_samples)
    
    # 4. 检测设备
    print(f"\n3. 检测训练设备")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"   - 设备: {device}")
    print(f"   - 数据类型: {dtype}")
    
    # 5. 加载模型和分词器
    # 正确的本地模型路径是snapshots下的具体版本目录
    actual_model_path = "/Users/Williamhiler/Documents/my-project/train/models/cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    print(f"\n4. 加载基础模型和分词器：{actual_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_path,
        use_fast=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=dtype,
        device_map=None if device == "cpu" else "auto",
        local_files_only=True
    )
    
    # 6. 加载第一步训练的LoRA权重
    print(f"\n5. 加载第一步LoRA权重：{Config.lora_weights}")
    if os.path.exists(Config.lora_weights):
        model = PeftModel.from_pretrained(
            model, 
            Config.lora_weights,
            torch_dtype=dtype,
            local_files_only=True
        )
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
    
    # 8. 配置训练器
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
            logging_steps=Config.logging_steps,
            optim="adamw_torch" if device == "cpu" else "adamw_torch",
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
    print(f"   - 训练设备: {device}")
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
    print("=" * 60)

if __name__ == "__main__":
    main()
