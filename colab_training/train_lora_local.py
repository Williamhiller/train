#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab训练入口脚本 - 使用Unsloth框架微调Qwen模型生成LoRA权重

功能：
1. 使用Unsloth框架加载Qwen2.5-2B-Instruct模型
2. 应用LoRA微调技术，显存占用低
3. 支持从JSON文件读取训练数据
4. 自动保存LoRA权重到out目录
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# ==================== 配置参数 ====================
class Config:
    # 模型配置
    model_name = "unsloth/Qwen2.5-2B-Instruct-bnb-4bit"  # 4-bit量化的Qwen模型，适合Colab
    max_seq_length = 2048  # 最大序列长度
    
    # LoRA配置
    lora_r = 32  # LoRA秩
    lora_alpha = 64  # LoRA alpha值
    lora_dropout = 0.1  # LoRA dropout比例
    
    # 训练数据配置
    data_path = "/content/my-project/train/v5/data/expert_data/qwen_finetune_data.json"  # Colab上的数据路径
    
    # 训练参数
    per_device_train_batch_size = 4  # 每个设备的训练批次大小
    gradient_accumulation_steps = 4  # 梯度累积步数
    warmup_steps = 100  # 预热步数
    max_steps = 500  # 最大训练步数
    learning_rate = 2e-5  # 学习率
    weight_decay = 0.01  # 权重衰减
    
    # 输出配置
    output_dir = "/content/my-project/train/colab_training/out"  # Colab上的输出目录
    logging_steps = 10  # 日志打印步数

# ==================== 数据处理 ====================
def format_training_data(examples):
    """格式化训练数据为Qwen模型所需的格式"""
    output_texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        # Qwen模型的聊天模板
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        output_texts.append(text)
    return {"text": output_texts}

# ==================== 主训练函数 ====================
def main():
    print("=" * 60)
    print("Colab Qwen模型LoRA微调训练开始")
    print("=" * 60)
    
    # 1. 加载基础模型
    print(f"\n1. 加载基础模型: {Config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.model_name,
        max_seq_length=Config.max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    
    # 2. 配置LoRA
    print(f"\n2. 配置LoRA参数")
    print(f"   - LoRA秩: {Config.lora_r}")
    print(f"   - LoRA Alpha: {Config.lora_alpha}")
    print(f"   - LoRA Dropout: {Config.lora_dropout}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=Config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # 3. 加载和格式化训练数据
    print(f"\n3. 加载训练数据: {Config.data_path}")
    dataset = load_dataset("json", data_files=Config.data_path, split="train")
    dataset = dataset.map(format_training_data, batched=True)
    print(f"   - 加载数据条数: {len(dataset)}")
    
    # 4. 配置训练器
    print(f"\n4. 配置训练器")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=Config.max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=Config.per_device_train_batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            warmup_steps=Config.warmup_steps,
            max_steps=Config.max_steps,
            learning_rate=Config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=Config.logging_steps,
            optim="adamw_8bit",
            weight_decay=Config.weight_decay,
            lr_scheduler_type="cosine",
            output_dir=Config.output_dir,
            report_to="none",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
        ),
    )
    
    # 5. 开始训练
    print(f"\n5. 开始训练")
    print(f"   - 训练设备: {torch.cuda.get_device_name()}")
    print(f"   - 训练批次大小: {Config.per_device_train_batch_size}")
    print(f"   - 梯度累积步数: {Config.gradient_accumulation_steps}")
    print(f"   - 最大训练步数: {Config.max_steps}")
    print("\n" + "=" * 60)
    
    trainer.train()
    
    # 6. 保存LoRA权重
    print(f"\n6. 保存LoRA权重到: {Config.output_dir}")
    final_output_dir = os.path.join(Config.output_dir, "qwen_lora_final")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\n" + "=" * 60)
    print("训练完成！")
    print(f"LoRA权重已保存到: {final_output_dir}")
    print("权重文件列表:")
    for file in os.listdir(final_output_dir):
        file_path = os.path.join(final_output_dir, file)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"   - {file}: {file_size:.2f} MB")
    print("=" * 60)

# ==================== 执行训练 ====================
if __name__ == "__main__":
    main()
