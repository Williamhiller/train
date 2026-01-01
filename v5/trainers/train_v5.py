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
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import V5FusionModel
from utils.data_processing.data_loader import V5DataLoader
from utils.model_utils.evaluation import evaluate_model
from utils.model_utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志
    
    Args:
        config: 配置字典
        
    Returns:
        日志记录器
    """
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
    """设置设备
    
    Returns:
        使用的设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def create_model(config: Dict, device: torch.device) -> V5FusionModel:
    """创建模型
    
    Args:
        config: 配置字典
        device: 设备
        
    Returns:
        模型实例
    """
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
    """创建优化器和学习率调度器
    
    Args:
        model: 模型
        config: 配置字典
        train_loader: 训练数据加载器
        
    Returns:
        优化器和学习率调度器
    """
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"]
    )
    
    # 计算总训练步数
    total_steps = len(train_loader) * config["training"]["epochs"] // config["training"]["gradient_accumulation_steps"]
    
    # 创建学习率调度器
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
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        config: 配置字典
        epoch: 当前epoch
        
    Returns:
        平均损失和准确率
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 获取数据
        structured_features = batch["structured_features"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text_features"]
        
        # 前向传播
        outputs = model(texts, structured_features)
        logits = outputs["logits"]
        
        # 计算损失
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # 反向传播
        loss = loss / config["training"]["gradient_accumulation_steps"]
        loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * config["training"]["gradient_accumulation_steps"]
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            "loss": loss.item() * config["training"]["gradient_accumulation_steps"],
            "acc": total_correct / total_samples
        })
        
        # 记录日志
        if (batch_idx + 1) % config["training"]["logging_steps"] == 0:
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def validate_epoch(model: V5FusionModel, 
                  val_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, float, Dict]:
    """验证一个epoch
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        
    Returns:
        平均损失、准确率和详细指标
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # 获取数据
            structured_features = batch["structured_features"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text_features"]
            
            # 前向传播
            outputs = model(texts, structured_features)
            logits = outputs["logits"]
            
            # 计算损失
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
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


def train_model(config: Dict) -> V5FusionModel:
    """训练模型
    
    Args:
        config: 配置字典
        
    Returns:
        训练好的模型
    """
    # 设置日志和设备
    logger = setup_logging(config)
    device = setup_device()
    
    logger.info("Starting model training")
    logger.info(f"Configuration: {config}")
    
    # 准备数据
    logger.info("Preparing data...")
    data_loader = V5DataLoader(config)
    
    # 获取用于限制训练数据量的batch_size参数（如果配置了的话）
    train_batch_size = config.get("training", {}).get("train_batch_size", None)
    logger.info(f"Using train_batch_size: {train_batch_size}")
    
    # 准备训练和验证数据
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
    
    # 检查是否存在检查点，如果存在则加载并继续训练
    start_epoch = 0
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    
    checkpoint_dir = config["saving"]["checkpoint_dir"]
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        try:
            # 加载检查点
            start_epoch, _ = load_checkpoint(model, optimizer, scheduler, latest_checkpoint, device)
            start_epoch += 1  # 从下一个epoch开始训练
            logger.info(f"Resuming training from epoch {start_epoch}")
            
            # 加载最佳F1值和早停计数器（如果有）
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
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch
        )
        
        # 验证
        val_loss, val_acc, val_metrics = validate_epoch(model, val_loader, device)
        
        # 记录日志
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # 保存检查点
        if (epoch + 1) % config["training"]["save_steps"] == 0:
            checkpoint_path = os.path.join(
                config["saving"]["checkpoint_dir"], 
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            # 添加额外信息到检查点
            additional_info = {
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement
            }
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path, additional_info)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            
            best_model_path = os.path.join(
                config["saving"]["checkpoint_dir"], 
                "best_model.pt"
            )
            # 添加额外信息到最佳模型检查点
            additional_info = {
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement
            }
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_model_path, additional_info)
            logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        # 早停
        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    logger.info("Training completed")
    return model


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train V5 Fusion Model")
    parser.add_argument("--config", type=str, default="configs/v5_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 训练模型
    model = train_model(config)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()