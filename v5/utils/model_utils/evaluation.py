import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from typing import Dict, List, Tuple
import logging


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        包含各种指标的字典
    """
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # 计算各类别指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算各类别准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "class_accuracies": class_accuracies.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["win", "draw", "loss"])
    }


def evaluate_model(model, data_loader, device: torch.device) -> Dict:
    """评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            structured_features = batch["structured_features"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text_features"]
            
            # 前向传播
            outputs = model(texts, structured_features)
            logits = outputs["logits"]
            probabilities = outputs["probabilities"]
            
            # 计算损失
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            
            # 获取预测
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / len(data_loader)
    
    # 添加概率统计
    probabilities_array = np.array(all_probabilities)
    metrics["probability_stats"] = {
        "mean_confidence": np.mean(np.max(probabilities_array, axis=1)),
        "std_confidence": np.std(np.max(probabilities_array, axis=1)),
        "mean_win_prob": np.mean(probabilities_array[:, 0]),
        "mean_draw_prob": np.mean(probabilities_array[:, 1]),
        "mean_loss_prob": np.mean(probabilities_array[:, 2])
    }
    
    return metrics


def evaluate_by_odds_range(model, data_loader, device: torch.device, odds_ranges: List[Tuple[float, float]]) -> Dict:
    """按赔率范围评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        odds_ranges: 赔率范围列表 [(min_odds, max_odds), ...]
        
    Returns:
        按赔率范围分组的评估结果
    """
    model.eval()
    
    # 初始化结果存储
    results = {}
    for i, (min_odds, max_odds) in enumerate(odds_ranges):
        results[f"range_{i}"] = {
            "min_odds": min_odds,
            "max_odds": max_odds,
            "predictions": [],
            "probabilities": [],
            "labels": []
        }
    
    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            structured_features = batch["structured_features"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text_features"]
            
            # 获取主胜赔率（假设在结构化特征的第0个位置）
            home_win_odds = structured_features[:, 0].cpu().numpy()
            
            # 前向传播
            outputs = model(texts, structured_features)
            logits = outputs["logits"]
            probabilities = outputs["probabilities"]
            
            # 获取预测
            predictions = torch.argmax(logits, dim=-1)
            
            # 按赔率范围分组
            for i, (min_odds, max_odds) in enumerate(odds_ranges):
                mask = (home_win_odds >= min_odds) & (home_win_odds < max_odds)
                
                if np.any(mask):
                    results[f"range_{i}"]["predictions"].extend(predictions[mask].cpu().numpy())
                    results[f"range_{i}"]["probabilities"].extend(probabilities[mask].cpu().numpy())
                    results[f"range_{i}"]["labels"].extend(labels[mask].cpu().numpy())
    
    # 计算各范围的指标
    for range_key, range_data in results.items():
        if range_data["labels"]:  # 确保有数据
            metrics = calculate_metrics(range_data["labels"], range_data["predictions"])
            results[range_key].update(metrics)
            results[range_key]["sample_count"] = len(range_data["labels"])
        else:
            results[range_key]["sample_count"] = 0
    
    return results


def evaluate_betting_performance(model, data_loader, device: torch.device, 
                               betting_strategy: str = "best_value") -> Dict:
    """评估投注性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        betting_strategy: 投注策略
        
    Returns:
        投注性能结果
    """
    model.eval()
    
    total_bet = 0
    total_return = 0
    winning_bets = 0
    total_bets = 0
    
    bet_details = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 获取数据
            structured_features = batch["structured_features"].to(device)
            labels = batch["label"].to(device)
            texts = batch["text_features"]
            
            # 获取赔率（假设在结构化特征的前3个位置）
            odds = structured_features[:, :3].cpu().numpy()  # [home_win, draw, away_win]
            
            # 前向传播
            outputs = model(texts, structured_features)
            probabilities = outputs["probabilities"].cpu().numpy()
            
            # 根据策略进行投注
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_probs = probabilities[i]
                match_odds = odds[i]
                
                # 决定是否投注
                if betting_strategy == "best_value":
                    # 选择期望价值最高的结果
                    expected_values = pred_probs * match_odds
                    best_bet = np.argmax(expected_values)
                    
                    # 只有当期望价值大于1时才投注
                    if expected_values[best_bet] > 1.0:
                        bet_amount = 1.0  # 单位投注
                        total_bet += bet_amount
                        total_bets += 1
                        
                        # 计算回报
                        if best_bet == true_label:
                            return_amount = bet_amount * match_odds[best_bet]
                            total_return += return_amount
                            winning_bets += 1
                            
                            bet_details.append({
                                "predicted": best_bet,
                                "actual": true_label,
                                "odds": match_odds[best_bet],
                                "probability": pred_probs[best_bet],
                                "expected_value": expected_values[best_bet],
                                "won": True,
                                "return": return_amount
                            })
                        else:
                            bet_details.append({
                                "predicted": best_bet,
                                "actual": true_label,
                                "odds": match_odds[best_bet],
                                "probability": pred_probs[best_bet],
                                "expected_value": expected_values[best_bet],
                                "won": False,
                                "return": 0
                            })
    
    # 计算投注性能指标
    if total_bets > 0:
        win_rate = winning_bets / total_bets
        roi = (total_return - total_bet) / total_bet * 100
        avg_odds = np.mean([detail["odds"] for detail in bet_details])
        avg_expected_value = np.mean([detail["expected_value"] for detail in bet_details])
    else:
        win_rate = 0
        roi = 0
        avg_odds = 0
        avg_expected_value = 0
    
    return {
        "total_bets": total_bets,
        "winning_bets": winning_bets,
        "win_rate": win_rate,
        "total_bet_amount": total_bet,
        "total_return": total_return,
        "roi": roi,
        "avg_odds": avg_odds,
        "avg_expected_value": avg_expected_value,
        "bet_details": bet_details
    }


def compare_models(model1, model2, data_loader, device: torch.device, 
                  model1_name: str = "Model 1", model2_name: str = "Model 2") -> Dict:
    """比较两个模型的性能
    
    Args:
        model1: 第一个模型
        model2: 第二个模型
        data_loader: 数据加载器
        device: 设备
        model1_name: 第一个模型名称
        model2_name: 第二个模型名称
        
    Returns:
        模型比较结果
    """
    # 评估第一个模型
    metrics1 = evaluate_model(model1, data_loader, device)
    
    # 评估第二个模型
    metrics2 = evaluate_model(model2, data_loader, device)
    
    # 计算差异
    comparison = {
        model1_name: metrics1,
        model2_name: metrics2,
        "differences": {
            "accuracy": metrics1["accuracy"] - metrics2["accuracy"],
            "f1": metrics1["f1"] - metrics2["f1"],
            "precision": metrics1["precision"] - metrics2["precision"],
            "recall": metrics1["recall"] - metrics2["recall"]
        }
    }
    
    # 进行统计显著性检验（简化版）
    comparison["significance"] = {
        "accuracy_better": metrics1["accuracy"] > metrics2["accuracy"],
        "f1_better": metrics1["f1"] > metrics2["f1"],
        "precision_better": metrics1["precision"] > metrics2["precision"],
        "recall_better": metrics1["recall"] > metrics2["recall"]
    }
    
    return comparison


def log_evaluation_results(metrics: Dict, logger: logging.Logger, prefix: str = ""):
    """记录评估结果
    
    Args:
        metrics: 评估指标
        logger: 日志记录器
        prefix: 日志前缀
    """
    if prefix:
        logger.info(f"{prefix} Evaluation Results:")
    else:
        logger.info("Evaluation Results:")
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    
    if "probability_stats" in metrics:
        stats = metrics["probability_stats"]
        logger.info(f"Mean Confidence: {stats['mean_confidence']:.4f}")
        logger.info(f"Mean Win Probability: {stats['mean_win_prob']:.4f}")
        logger.info(f"Mean Draw Probability: {stats['mean_draw_prob']:.4f}")
        logger.info(f"Mean Loss Probability: {stats['mean_loss_prob']:.4f}")
    
    if "roi" in metrics:
        logger.info(f"ROI: {metrics['roi']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']:.4f}")
        logger.info(f"Total Bets: {metrics['total_bets']}")