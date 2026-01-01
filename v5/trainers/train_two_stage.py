#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
两阶段训练脚本
阶段1：预训练，让Qwen学习专家知识
阶段2：微调，将专家知识应用到比赛预测
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
from datetime import datetime
import argparse
from typing import Dict, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import V5FusionModel
from utils.data_processing.data_loader import V5DataLoader
from utils.model_utils.evaluation import evaluate_model
from utils.model_utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.expert_knowledge.expert_knowledge_extractor import ExpertKnowledgeExtractor


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志"""
    log_dir = os.path.dirname(config["logging"]["file"])
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
        handlers=[
            logging.FileHandler(config["logging"]["file"]),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def setup_device() -> torch.device:
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def create_model(config: Dict, device: torch.device) -> V5FusionModel:
    """创建模型"""
    # 获取结构化数据输入大小
    from utils.data_processing.data_loader import V5DataLoader
    data_loader = V5DataLoader(config)
    feature_info = data_loader.get_feature_info(config["data"]["raw_data_path"])
    
    # 更新配置
    config["structured_input_size"] = feature_info["num_features"]
    
    # 创建模型
    model = V5FusionModel(config)
    model.to(device)
    
    return model


def create_optimizer_and_scheduler(model: V5FusionModel, config: Dict, train_loader: DataLoader) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    """创建优化器和学习率调度器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"]
    )
    
    total_steps = len(train_loader) * config["training"]["epochs"] // config["training"]["gradient_accumulation_steps"]
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def train_epoch(model: V5FusionModel, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                scheduler: optim.lr_scheduler.LambdaLR,
                device: torch.device, 
                config: Dict,
                epoch: int) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        structured_features = batch["structured_features"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text_features"]
        
        outputs = model(texts, structured_features)
        logits = outputs["logits"]
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        loss = loss / config["training"]["gradient_accumulation_steps"]
        loss.backward()
        
        if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config["training"]["gradient_accumulation_steps"]
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        progress_bar.set_postfix({
            "loss": loss.item() * config["training"]["gradient_accumulation_steps"],
            "acc": total_correct / total_samples
        })
        
        if (batch_idx + 1) % config["training"]["logging_steps"] == 0:
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def validate_epoch(model: V5FusionModel, 
                  val_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, float, Dict]:
    """验证一个epoch"""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            structured_features = batch["structured_features"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text_features"]
            
            outputs = model(texts, structured_features)
            logits = outputs["logits"]
            
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": classification_report(all_labels, all_predictions, target_names=["win", "draw", "loss"])
    }
    
    return avg_loss, accuracy, metrics


def stage1_pretrain_expert_knowledge(config: Dict, device: torch.device):
    """阶段1：预训练专家知识
    
    Args:
        config: 配置字典
        device: 设备
    """
    print("\n" + "="*80)
    print("阶段1：预训练专家知识")
    print("="*80)
    
    logger = setup_logging(config)
    
    # 提取专家知识
    print("\n提取专家知识...")
    extractor = ExpertKnowledgeExtractor(config)
    expert_data_path = "data/expert_data/expert_training_data.json"
    rules = extractor.extract_rules_from_expert_data(expert_data_path)
    
    # 保存规则
    rules_path = "data/expert_knowledge/expert_rules.json"
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)
    extractor.save_rules(rules_path)
    
    print("\n✅ 阶段1完成：专家知识已提取并保存")
    print(f"   规则数量: {len(rules)}")
    print(f"   规则保存路径: {rules_path}")
    
    return True


def stage2_fine_tune_with_matches(config: Dict, device: torch.device):
    """阶段2：使用比赛数据微调（带专家知识）
    
    Args:
        config: 配置字典
        device: 设备
    """
    print("\n" + "="*80)
    print("阶段2：微调比赛预测（带专家知识）")
    print("="*80)
    
    logger = setup_logging(config)
    
    logger.info("Starting stage 2: Fine-tuning with expert knowledge")
    logger.info(f"Configuration: {config}")
    
    # 准备数据
    logger.info("Preparing data...")
    data_loader = V5DataLoader(config)
    
    train_batch_size = config.get("training", {}).get("train_batch_size", None)
    logger.info(f"Using train_batch_size: {train_batch_size}")
    
    train_loader, val_loader = data_loader.prepare_data(
        config["data"]["raw_data_path"],
        config["data"]["expert_data_path"],
        batch_size=train_batch_size
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(config, device)
    
    # 创建优化器和调度器
    logger.info("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, train_loader)
    
    # 检查检查点
    start_epoch = 0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    
    checkpoint_dir = config["saving"]["checkpoint_dir"]
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        try:
            start_epoch, _ = load_checkpoint(model, optimizer, scheduler, latest_checkpoint, device)
            start_epoch += 1
            logger.info(f"Resuming training from epoch {start_epoch}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            if "best_val_f1" in checkpoint:
                best_val_f1 = checkpoint["best_val_f1"]
            if "epochs_without_improvement" in checkpoint:
                epochs_without_improvement = checkpoint["epochs_without_improvement"]
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    # 训练循环
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch
        )
        
        val_loss, val_acc, val_metrics = validate_epoch(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if (epoch + 1) % config["training"]["save_steps"] == 0:
            checkpoint_path = os.path.join(
                config["saving"]["checkpoint_dir"], 
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            additional_info = {
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement
            }
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path, additional_info)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            
            best_model_path = os.path.join(
                config["saving"]["checkpoint_dir"], 
                "best_model.pt"
            )
            additional_info = {
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement
            }
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_model_path, additional_info)
            logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    logger.info("Stage 2 training completed")
    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Two-Stage Training for V5 Fusion Model")
    parser.add_argument("--config", type=str, default="configs/v5_config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"], help="Training stage: 1 (expert knowledge), 2 (fine-tuning), or both")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = setup_device()
    
    # 执行训练阶段
    if args.stage in ["1", "both"]:
        stage1_pretrain_expert_knowledge(config, device)
    
    if args.stage in ["2", "both"]:
        stage2_fine_tune_with_matches(config, device)
    
    print("\n" + "="*80)
    print("所有训练阶段完成！")
    print("="*80)


if __name__ == "__main__":
    main()
