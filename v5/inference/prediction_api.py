import os
import sys
import yaml
import torch
import argparse
from typing import Dict, List, Optional, Any
import json
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import V5FusionModel
from utils.data_processing.match_data_processor import MatchDataProcessor
from utils.data_processing.expert_data_processor import ExpertDataProcessor


class PredictionAPI:
    """预测API服务"""
    
    def __init__(self, config_path: str, model_path: str = None):
        """初始化API服务
        
        Args:
            config_path: 配置文件路径
            model_path: 模型路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化数据处理器
        self.match_processor = MatchDataProcessor(self.config)
        self.expert_processor = ExpertDataProcessor(self.config)
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(self.config["saving"]["checkpoint_dir"], "best_model.pt")
        
        self.model = self._load_model(model_path)
        
        # 加载专家数据
        expert_data_path = self.config["data"]["expert_data_path"]
        if os.path.exists(expert_data_path):
            with open(expert_data_path, 'r', encoding='utf-8') as f:
                self.expert_data = json.load(f)
        else:
            self.expert_data = {}
        
        # 创建Flask应用
        self.app = Flask(__name__)
        self._setup_routes()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str) -> V5FusionModel:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        model_config = checkpoint.get("config", self.config)
        
        # 创建模型
        model = V5FusionModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded from {model_path}")
        return model
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_version": self.config["model"]["version"]
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """预测比赛结果"""
            try:
                # 获取请求数据
                data = request.get_json()
                
                # 验证必需字段
                required_fields = ["home_team", "away_team"]
                for field in required_fields:
                    if field not in data:
                        return jsonify({
                            "error": f"Missing required field: {field}"
                        }), 400
                
                # 解析数据
                parsed_data = self._parse_request_data(data)
                
                # 创建特征
                structured_features = self._create_structured_features(parsed_data)
                text_features = self._create_text_features(parsed_data)
                
                # 模型预测
                with torch.no_grad():
                    structured_tensor = torch.tensor(structured_features, dtype=torch.float).unsqueeze(0).to(self.device)
                    outputs = self.model([text_features], structured_tensor)
                    probabilities = outputs["probabilities"].cpu().numpy()[0]
                    logits = outputs["logits"].cpu().numpy()[0]
                
                # 解析结果
                result_idx = np.argmax(probabilities)
                result_map = ["home_win", "draw", "away_win"]
                predicted_result = result_map[result_idx]
                
                # 构建响应
                response = {
                    "predicted_result": predicted_result,
                    "probabilities": {
                        "home_win": float(probabilities[0]),
                        "draw": float(probabilities[1]),
                        "away_win": float(probabilities[2])
                    },
                    "confidence": float(np.max(probabilities)),
                    "request_id": data.get("request_id", ""),
                    "timestamp": datetime.now().isoformat()
                }
                
                # 添加投注建议（如果有赔率信息）
                if all(key in data for key in ["home_win_odds", "draw_odds", "away_win_odds"]):
                    home_odds = data["home_win_odds"]
                    draw_odds = data["draw_odds"]
                    away_odds = data["away_win_odds"]
                    
                    # 计算期望价值
                    home_ev = probabilities[0] * home_odds
                    draw_ev = probabilities[1] * draw_odds
                    away_ev = probabilities[2] * away_odds
                    
                    ev_map = {"home_win": home_ev, "draw": draw_ev, "away_win": away_ev}
                    best_bet = max(ev_map, key=ev_map.get)
                    best_ev = ev_map[best_bet]
                    
                    response["betting_advice"] = {
                        "best_bet": best_bet,
                        "expected_value": {
                            "home_win": float(home_ev),
                            "draw": float(draw_ev),
                            "away_win": float(away_ev)
                        },
                        "best_expected_value": float(best_ev),
                        "has_value": best_ev > 1.0
                    }
                
                return jsonify(response)
            
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                return jsonify({
                    "error": str(e)
                }), 500
        
        @self.app.route('/predict_batch', methods=['POST'])
        def predict_batch():
            """批量预测比赛结果"""
            try:
                # 获取请求数据
                data = request.get_json()
                
                # 验证必需字段
                if "matches" not in data:
                    return jsonify({
                        "error": "Missing required field: matches"
                    }), 400
                
                matches = data["matches"]
                results = []
                
                # 批量处理
                for match in matches:
                    try:
                        # 验证必需字段
                        required_fields = ["home_team", "away_team"]
                        for field in required_fields:
                            if field not in match:
                                results.append({
                                    "error": f"Missing required field: {field}",
                                    "match": match
                                })
                                continue
                        
                        # 解析数据
                        parsed_data = self._parse_request_data(match)
                        
                        # 创建特征
                        structured_features = self._create_structured_features(parsed_data)
                        text_features = self._create_text_features(parsed_data)
                        
                        # 模型预测
                        with torch.no_grad():
                            structured_tensor = torch.tensor(structured_features, dtype=torch.float).unsqueeze(0).to(self.device)
                            outputs = self.model([text_features], structured_tensor)
                            probabilities = outputs["probabilities"].cpu().numpy()[0]
                        
                        # 解析结果
                        result_idx = np.argmax(probabilities)
                        result_map = ["home_win", "draw", "away_win"]
                        predicted_result = result_map[result_idx]
                        
                        # 构建响应
                        result = {
                            "predicted_result": predicted_result,
                            "probabilities": {
                                "home_win": float(probabilities[0]),
                                "draw": float(probabilities[1]),
                                "away_win": float(probabilities[2])
                            },
                            "confidence": float(np.max(probabilities))
                        }
                        
                        results.append(result)
                    
                    except Exception as e:
                        self.logger.error(f"Batch prediction error for match {match}: {str(e)}")
                        results.append({
                            "error": str(e),
                            "match": match
                        })
                
                return jsonify({
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({
                    "error": str(e)
                }), 500
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """获取模型信息"""
            return jsonify({
                "model_name": self.config["model"]["name"],
                "model_version": self.config["model"]["version"],
                "description": self.config["model"]["description"],
                "timestamp": datetime.now().isoformat()
            })
    
    def _parse_request_data(self, data: Dict) -> Dict:
        """解析请求数据
        
        Args:
            data: 请求数据
            
        Returns:
            解析后的数据字典
        """
        parsed_data = {
            "home_team": data.get("home_team", ""),
            "away_team": data.get("away_team", ""),
            "home_form": data.get("home_form", ""),
            "away_form": data.get("away_form", ""),
            "home_odds": data.get("home_win_odds", 0.0),
            "draw_odds": data.get("draw_odds", 0.0),
            "away_odds": data.get("away_win_odds", 0.0),
            "additional_info": data.get("additional_info", "")
        }
        
        return parsed_data
    
    def _create_structured_features(self, parsed_data: Dict) -> np.ndarray:
        """创建结构化特征
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            结构化特征数组
        """
        # 创建默认特征（简化版）
        features = np.zeros(23)  # 与训练时的特征数量一致
        
        # 填充赔率特征
        features[0] = parsed_data["home_odds"] if parsed_data["home_odds"] > 0 else 2.5  # home_win_odds
        features[1] = parsed_data["draw_odds"] if parsed_data["draw_odds"] > 0 else 3.2   # draw_odds
        features[2] = parsed_data["away_odds"] if parsed_data["away_odds"] > 0 else 2.8   # away_win_odds
        
        # 计算隐含概率
        if features[0] > 0 and features[1] > 0 and features[2] > 0:
            total_implied = 1.0/features[0] + 1.0/features[1] + 1.0/features[2]
            features[3] = (1.0/features[0]) / total_implied  # implied_home_win
            features[4] = (1.0/features[1]) / total_implied  # implied_draw
            features[5] = (1.0/features[2]) / total_implied  # implied_away_win
        
        # 填充其他特征（使用默认值）
        features[6] = 2.0  # home_wins
        features[7] = 1.0  # home_draws
        features[8] = 1.0  # home_losses
        features[9] = 5.0  # home_goals_scored
        features[10] = 3.0  # home_goals_conceded
        features[11] = 2.0  # home_goal_difference
        features[12] = 7.0  # home_form_points
        features[13] = 4.0  # home_matches_played
        
        features[14] = 1.0  # away_wins
        features[15] = 2.0  # away_draws
        features[16] = 2.0  # away_losses
        features[17] = 4.0  # away_goals_scored
        features[18] = 4.0  # away_goals_conceded
        features[19] = 0.0  # away_goal_difference
        features[20] = 5.0  # away_form_points
        features[21] = 4.0  # away_matches_played
        
        features[22] = 2.5  # over_under_odds
        
        # 标准化特征（使用训练时的标准化器）
        # 这里简化处理，实际应用中应该保存训练时的标准化器
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def _create_text_features(self, parsed_data: Dict) -> str:
        """创建文本特征
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            文本特征字符串
        """
        # 构建文本特征
        text_parts = []
        
        # 球队信息
        if parsed_data["home_team"] and parsed_data["away_team"]:
            text_parts.append(f"{parsed_data['home_team']}主场对阵{parsed_data['away_team']}客场")
        
        # 状态信息
        if parsed_data["home_form"]:
            text_parts.append(f"主队状态：{parsed_data['home_form']}")
        
        if parsed_data["away_form"]:
            text_parts.append(f"客队状态：{parsed_data['away_form']}")
        
        # 赔率信息
        if parsed_data["home_odds"] > 0 and parsed_data["draw_odds"] > 0 and parsed_data["away_odds"] > 0:
            text_parts.append(f"主胜赔率{parsed_data['home_odds']}，平局赔率{parsed_data['draw_odds']}，客胜赔率{parsed_data['away_odds']}")
        
        # 附加信息
        if parsed_data["additional_info"]:
            text_parts.append(f"其他信息：{parsed_data['additional_info']}")
        
        # 添加专家分析（如果有）
        if self.expert_data:
            # 简化处理，随机选择一些专家规则
            all_rules = []
            for pdf_name, pdf_data in self.expert_data.items():
                for rule_type, rules in pdf_data["rules"].items():
                    all_rules.extend(rules[:2])  # 每种类型取前2条规则
            
            if all_rules:
                selected_rules = np.random.choice(all_rules, min(3, len(all_rules)), replace=False)
                text_parts.append(f"专家分析：{'。'.join(selected_rules)}")
        
        return "。".join(text_parts)
    
    def run(self, host: str = None, port: int = None, debug: bool = False):
        """运行API服务
        
        Args:
            host: 主机地址
            port: 端口号
            debug: 调试模式
        """
        host = host or self.config["api"]["host"]
        port = port or self.config["api"]["port"]
        debug = debug or self.config["api"]["debug"]
        
        self.logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V5 Model Prediction API")
    parser.add_argument("--config", type=str, default="configs/v5_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--host", type=str, default=None, help="Host address")
    parser.add_argument("--port", type=int, default=None, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    # 创建API服务
    api = PredictionAPI(args.config, args.model)
    
    # 运行API服务
    api.run(args.host, args.port, args.debug)


if __name__ == "__main__":
    main()