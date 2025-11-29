#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛预测模型训练脚本

使用方法：
    python train.py --config config/config.yaml
"""

import argparse
import os
import sys
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import yaml
from utils.model_utils import (
    load_pretrained_model,
    setup_peft_model,
    save_model,
    format_training_prompt
)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='足球比赛预测模型训练')
    parser.add_argument('--config', type=str, 
                        default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume_from_checkpoint', type=str,
                        default=None,
                        help='从检查点恢复训练')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    加载配置文件
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        sys.exit(1)

def load_and_prepare_dataset(train_file: str, val_file: str, tokenizer: AutoTokenizer, max_length: int = 1024):
    """
    加载和准备训练数据集
    
    Args:
        train_file: 训练数据文件路径
        val_file: 验证数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        准备好的数据集
    """
    def preprocess_function(examples):
        """预处理单个样本"""
        # 格式化提示和回答
        formatted_texts = []
        for input_text, output_text in zip(examples['input'], examples['output']):
            formatted_text = format_training_prompt(input_text, output_text)
            formatted_texts.append(formatted_text)
        
        # 分词
        tokenized = tokenizer(
            formatted_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 设置标签（与输入相同，但mask掉padding部分）
        tokenized['labels'] = tokenized['input_ids'].clone()
        if tokenizer.pad_token_id is not None:
            tokenized['labels'][tokenized['input_ids'] == tokenizer.pad_token_id] = -100
        
        return tokenized
    
    # 加载数据集
    dataset = load_dataset(
        'json',
        data_files={
            'train': train_file,
            'validation': val_file
        }
    )
    
    # 预处理数据集
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset

def setup_training_args(config: dict) -> TrainingArguments:
    """
    设置训练参数
    
    Args:
        config: 配置字典
        
    Returns:
        训练参数对象
    """
    train_config = config['training']
    
    # 创建输出目录
    os.makedirs(train_config['output_dir'], exist_ok=True)
    
    # 配置日志目录
    logging_dir = os.path.join(train_config['output_dir'], 'logs')
    os.makedirs(logging_dir, exist_ok=True)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        evaluation_strategy=train_config['evaluation_strategy'],
        eval_steps=train_config['eval_steps'],
        seed=train_config['seed'],
        fp16=train_config['fp16'] and torch.cuda.is_available(),
        push_to_hub=train_config['push_to_hub'],
        hub_model_id=train_config['hub_model_id'] if train_config['push_to_hub'] else None,
        logging_dir=logging_dir,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=3,  # 只保存最新的3个模型
        label_names=['labels']
    )
    
    return training_args

def train_model(config: dict, resume_from_checkpoint: str = None):
    """
    训练模型的主函数
    
    Args:
        config: 配置字典
        resume_from_checkpoint: 检查点路径
    """
    # 打印配置信息
    print("===== 训练配置 ====")
    print(f"模型名称: {config['model']['name']}")
    print(f"输出目录: {config['training']['output_dir']}")
    print(f"训练轮数: {config['training']['num_epochs']}")
    print(f"使用PEFT: {config['peft']['use_peft']}")
    print("==================")
    
    # 加载预训练模型和分词器
    model, tokenizer = load_pretrained_model(
        model_name=config['model']['name'],
        quantization=config['model']['quantization'],
        quant_bits=config['model']['quant_bits']
    )
    
    # 设置PEFT
    model = setup_peft_model(model, config['peft'])
    
    # 加载和准备数据集
    tokenized_dataset = load_and_prepare_dataset(
        train_file=config['data']['train_file'],
        val_file=config['data']['validation_file'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    # 设置数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型不需要掩码语言建模
    )
    
    # 设置训练参数
    training_args = setup_training_args(config)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 评估模型
    print("评估模型...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")
    
    # 保存最终模型
    final_model_path = os.path.join(config['training']['output_dir'], 'final')
    save_model(
        model=model,
        tokenizer=tokenizer,
        save_path=final_model_path,
        is_peft=config['peft']['use_peft']
    )
    
    # 保存评估结果
    with open(os.path.join(config['training']['output_dir'], 'eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    print("训练完成！")

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确保数据文件存在
    for data_file_key in ['train_file', 'validation_file']:
        data_file = config['data'].get(data_file_key)
        if not data_file or not os.path.exists(data_file):
            print(f"错误：数据文件 {data_file} 不存在")
            print("请先运行 preprocess.py 准备数据")
            sys.exit(1)
    
    # 开始训练
    train_model(config, args.resume_from_checkpoint)

if __name__ == "__main__":
    main()