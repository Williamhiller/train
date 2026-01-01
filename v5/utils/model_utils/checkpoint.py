import os
import torch
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LambdaLR,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   additional_info: Optional[Dict] = None):
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
        additional_info: 额外信息
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }
    
    # 添加额外信息
    if additional_info:
        checkpoint.update(additional_info)
    
    # 保存检查点
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LambdaLR,
                   filepath: str,
                   device: torch.device) -> Tuple[int, float]:
    """加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        filepath: 检查点路径
        device: 设备
        
    Returns:
        当前epoch和损失
    """
    # 加载检查点
    checkpoint = torch.load(filepath, map_location=device)
    
    # 加载状态
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    loss = checkpoint.get("loss", 0.0)
    
    logging.info(f"Checkpoint loaded from {filepath}, epoch: {epoch}, loss: {loss}")
    
    return epoch, loss


def save_model_only(model: torch.nn.Module, filepath: str):
    """仅保存模型参数
    
    Args:
        model: 模型
        filepath: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存模型参数
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")


def load_model_only(model: torch.nn.Module, filepath: str, device: torch.device):
    """仅加载模型参数
    
    Args:
        model: 模型
        filepath: 模型路径
        device: 设备
    """
    # 加载模型参数
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f"Model loaded from {filepath}")


def save_model_info(model: torch.nn.Module, config: Dict, filepath: str):
    """保存模型信息
    
    Args:
        model: 模型
        config: 配置
        filepath: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 准备模型信息
    model_info = {
        "model_type": "V5FusionModel",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "config": config,
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存模型信息
    with open(filepath, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logging.info(f"Model info saved to {filepath}")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """查找最新的检查点
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        最新检查点路径，如果没有则返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 获取所有检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # 返回最新的检查点
    return os.path.join(checkpoint_dir, checkpoint_files[0])


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 5):
    """清理旧检查点
    
    Args:
        checkpoint_dir: 检查点目录
        keep_last_n: 保留的检查点数量
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # 获取所有检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and f != 'best_model.pt']
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # 删除旧检查点
    for checkpoint_file in checkpoint_files[keep_last_n:]:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        os.remove(checkpoint_path)
        logging.info(f"Removed old checkpoint: {checkpoint_path}")


def export_model_for_inference(model: torch.nn.Module, 
                              tokenizer, 
                              config: Dict,
                              model_dir: str):
    """导出模型用于推理
    
    Args:
        model: 模型
        tokenizer: 分词器
        config: 配置
        model_dir: 导出目录
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, "model.pt")
    save_model_only(model, model_path)
    
    # 保存分词器
    if hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(model_dir)
    
    # 保存配置
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存模型信息
    info_path = os.path.join(model_dir, "model_info.json")
    save_model_info(model, config, info_path)
    
    logging.info(f"Model exported for inference to {model_dir}")


def create_model_card(model_dir: str, 
                     model_name: str,
                     description: str,
                     performance_metrics: Dict):
    """创建模型卡片
    
    Args:
        model_dir: 模型目录
        model_name: 模型名称
        description: 模型描述
        performance_metrics: 性能指标
    """
    # 模型卡片内容
    model_card = f"""# {model_name}

## 模型描述

{description}

## 性能指标

"""
    
    # 添加性能指标
    for metric_name, metric_value in performance_metrics.items():
        model_card += f"- {metric_name}: {metric_value:.4f}\n"
    
    # 添加使用方法
    model_card += """
## 使用方法

```python
from models.fusion_model import V5FusionModel
from transformers import AutoTokenizer

# 加载模型
model = V5FusionModel.from_pretrained("./")
tokenizer = AutoTokenizer.from_pretrained("./")

# 进行预测
texts = ["主队近期状态良好，客队表现不佳"]
structured_data = torch.tensor([...])  # 结构化数据

outputs = model(texts, structured_data)
predictions = torch.argmax(outputs["logits"], dim=-1)
```
"""
    
    # 保存模型卡片
    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, 'w') as f:
        f.write(model_card)
    
    logging.info(f"Model card created at {card_path}")