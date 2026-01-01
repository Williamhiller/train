import os
import sys
import yaml
import torch
import argparse
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import V5FusionModel
from utils.data_processing.match_data_processor import MatchDataProcessor
from utils.data_processing.expert_data_processor import ExpertDataProcessor
from utils.feature_engineering.feature_engineer import FeatureEngineer


class MatchPredictor:
    """比赛预测器"""
    
    def __init__(self, config_path: str, model_path: str = None):
        """初始化预测器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型路径（如果为None，则使用配置中的最佳模型）
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化数据处理器
        self.match_processor = MatchDataProcessor(self.config)
        self.expert_processor = ExpertDataProcessor(self.config)
        
        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer(self.config)
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(self.config["saving"]["checkpoint_dir"], "best_model.pt")
        
        self.model = self._load_model(model_path)
        
        # 加载专家数据
        expert_data_path = self.config["data"]["expert_data_path"]
        if os.path.exists(expert_data_path):
            import json
            with open(expert_data_path, 'r', encoding='utf-8') as f:
                self.expert_data = json.load(f)
        else:
            self.expert_data = {}
    
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
        
        return model
    
    def parse_user_input(self, user_input: str) -> Dict:
        """解析用户输入
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            解析后的数据字典
        """
        # 初始化结果
        parsed_data = {
            "home_team": "",
            "away_team": "",
            "home_form": "",
            "away_form": "",
            "home_odds": 0.0,
            "draw_odds": 0.0,
            "away_odds": 0.0,
            "additional_info": ""
        }
        
        # 提取球队名称
        team_pattern = r"([^\s,，]+队|[^\s,，]+)"
        teams = re.findall(team_pattern, user_input)
        
        if len(teams) >= 2:
            parsed_data["home_team"] = teams[0]
            parsed_data["away_team"] = teams[1]
        
        # 提取赔率
        odds_patterns = [
            r"主胜.*?(\d+\.?\d*)",
            r"平局.*?(\d+\.?\d*)",
            r"客胜.*?(\d+\.?\d*)",
            r"胜.*?(\d+\.?\d*).*?平.*?(\d+\.?\d*).*?胜.*?(\d+\.?\d*)",
            r"(\d+\.?\d*).*?(\d+\.?\d*).*?(\d+\.?\d*)"
        ]
        
        for pattern in odds_patterns:
            match = re.search(pattern, user_input)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    parsed_data["home_odds"] = float(groups[0])
                    parsed_data["draw_odds"] = float(groups[1])
                    parsed_data["away_odds"] = float(groups[2])
                    break
                elif len(groups) == 1:
                    # 单个赔率，需要根据上下文判断
                    if "主胜" in pattern:
                        parsed_data["home_odds"] = float(groups[0])
                    elif "平局" in pattern:
                        parsed_data["draw_odds"] = float(groups[0])
                    elif "客胜" in pattern:
                        parsed_data["away_odds"] = float(groups[0])
        
        # 提取状态信息
        form_patterns = [
            r"([^,，。]*?近.*?[^,，。]*)",
            r"([^,，。]*?状态.*?[^,，。]*)",
            r"([^,，。]*?表现.*?[^,，。]*)"
        ]
        
        for pattern in form_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                if "主队" in match or parsed_data["home_team"] in match:
                    parsed_data["home_form"] = match
                elif "客队" in match or parsed_data["away_team"] in match:
                    parsed_data["away_form"] = match
        
        # 保存原始输入作为附加信息
        parsed_data["additional_info"] = user_input
        
        return parsed_data
    
    def create_structured_features(self, parsed_data: Dict) -> np.ndarray:
        """创建结构化特征
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            结构化特征数组
        """
        # 创建基本特征字典
        basic_features = {
            "home_win_odds": parsed_data["home_odds"] if parsed_data["home_odds"] > 0 else 2.5,
            "draw_odds": parsed_data["draw_odds"] if parsed_data["draw_odds"] > 0 else 3.2,
            "away_win_odds": parsed_data["away_odds"] if parsed_data["away_odds"] > 0 else 2.8,
            "over_under_odds": 2.5,
            "home_wins": 2.0,
            "home_draws": 1.0,
            "home_losses": 1.0,
            "home_goals_scored": 5.0,
            "home_goals_conceded": 3.0,
            "home_goal_difference": 2.0,
            "home_form_points": 7.0,
            "home_matches_played": 4.0,
            "away_wins": 1.0,
            "away_draws": 2.0,
            "away_losses": 2.0,
            "away_goals_scored": 4.0,
            "away_goals_conceded": 4.0,
            "away_goal_difference": 0.0,
            "away_form_points": 5.0,
            "away_matches_played": 4.0
        }
        
        # 计算隐含概率
        home_odds = basic_features["home_win_odds"]
        draw_odds = basic_features["draw_odds"]
        away_odds = basic_features["away_win_odds"]
        
        if home_odds > 0 and draw_odds > 0 and away_odds > 0:
            total_implied = 1.0/home_odds + 1.0/draw_odds + 1.0/away_odds
            basic_features["implied_home_win"] = (1.0/home_odds) / total_implied
            basic_features["implied_draw"] = (1.0/draw_odds) / total_implied
            basic_features["implied_away_win"] = (1.0/away_odds) / total_implied
        else:
            basic_features["implied_home_win"] = 0.0
            basic_features["implied_draw"] = 0.0
            basic_features["implied_away_win"] = 0.0
        
        # 转换为DataFrame以便使用FeatureEngineer
        features_df = pd.DataFrame([basic_features])
        
        # 使用V3版本的特征提取器（如果可用）
        if self.feature_engineer.odds_extractor is not None and self.feature_engineer.team_state_extractor is not None:
            try:
                # 提取赔率特征
                odds_features = self.feature_engineer.odds_extractor.extract_odds_features(features_df)
                features_df = pd.concat([features_df, odds_features], axis=1)
                
                # 提取球队状态特征
                team_state_features = self.feature_engineer.team_state_extractor.extract_team_state_features(features_df)
                features_df = pd.concat([features_df, team_state_features], axis=1)
            except Exception as e:
                print(f"警告: 无法使用V3特征提取器: {e}")
        
        # 使用FeatureEngineer创建更丰富的特征
        features_df = self.feature_engineer.create_interaction_features(features_df)
        features_df = self.feature_engineer.create_ranking_features(features_df)
        features_df = self.feature_engineer.create_categorical_features(features_df)
        
        # 处理分类特征（独热编码）
        categorical_cols = features_df.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
        
        # 确保特征数量与训练时一致（23个特征）
        # 如果特征数量不足，填充0；如果过多，选择前23个
        final_features = np.zeros(23)
        
        # 选择数值特征
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        selected_features = features_df[numeric_features].values[0]
        
        # 填充特征
        n_features = min(len(selected_features), 23)
        final_features[:n_features] = selected_features[:n_features]
        
        # 标准化特征（使用训练时的标准化器）
        # 这里简化处理，实际应用中应该保存训练时的标准化器
        final_features = (final_features - np.mean(final_features)) / (np.std(final_features) + 1e-8)
        
        return final_features
    
    def create_text_features(self, parsed_data: Dict) -> str:
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
    
    def predict_match(self, user_input: str) -> Dict:
        """预测比赛结果
        
        Args:
            user_input: 用户输入
            
        Returns:
            预测结果字典
        """
        # 解析用户输入
        parsed_data = self.parse_user_input(user_input)
        
        # 创建特征
        structured_features = self.create_structured_features(parsed_data)
        text_features = self.create_text_features(parsed_data)
        
        # 转换为张量
        structured_tensor = torch.tensor(structured_features, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model([text_features], structured_tensor)
            probabilities = outputs["probabilities"].cpu().numpy()[0]
            logits = outputs["logits"].cpu().numpy()[0]
        
        # 解析结果
        result_idx = np.argmax(probabilities)
        result_map = ["主胜", "平局", "客胜"]
        predicted_result = result_map[result_idx]
        
        # 构建返回结果
        prediction_result = {
            "predicted_result": predicted_result,
            "probabilities": {
                "主胜": float(probabilities[0]),
                "平局": float(probabilities[1]),
                "客胜": float(probabilities[2])
            },
            "confidence": float(np.max(probabilities)),
            "parsed_data": parsed_data,
            "text_features": text_features,
            "structured_features": structured_features.tolist()
        }
        
        return prediction_result
    
    def generate_response(self, prediction_result: Dict) -> str:
        """生成响应文本
        
        Args:
            prediction_result: 预测结果
            
        Returns:
            响应文本
        """
        predicted_result_text = prediction_result["predicted_result"]
        confidence = prediction_result["confidence"]
        probabilities = prediction_result["probabilities"]
        
        # 构建响应
        response = f"根据分析，我预测这场比赛的结果是：{predicted_result_text}。\n\n"
        response += f"预测概率分布：\n"
        response += f"- 主胜：{probabilities['主胜']:.1%}\n"
        response += f"- 平局：{probabilities['平局']:.1%}\n"
        response += f"- 客胜：{probabilities['客胜']:.1%}\n\n"
        response += f"预测置信度：{confidence:.1%}\n\n"
        
        # 添加分析说明
        if confidence > 0.7:
            response += "这是一个高置信度的预测，模型认为这个结果的可能性较大。"
        elif confidence > 0.5:
            response += "这是一个中等置信度的预测，比赛结果存在一定不确定性。"
        else:
            response += "这是一个低置信度的预测，比赛结果难以确定，建议谨慎参考。"
        
        # 添加投注建议（如果有赔率信息）
        parsed_data = prediction_result["parsed_data"]
        if parsed_data["home_odds"] > 0 and parsed_data["draw_odds"] > 0 and parsed_data["away_odds"] > 0:
            response += "\n\n投注价值分析：\n"
            
            # 计算期望价值
            home_ev = probabilities["主胜"] * parsed_data["home_odds"]
            draw_ev = probabilities["平局"] * parsed_data["draw_odds"]
            away_ev = probabilities["客胜"] * parsed_data["away_odds"]
            
            ev_map = {"主胜": home_ev, "平局": draw_ev, "客胜": away_ev}
            best_bet = max(ev_map, key=ev_map.get)
            best_ev = ev_map[best_bet]
            
            if best_ev > 1.0:
                response += f"根据概率和赔率计算，{best_bet}的期望价值为{best_ev:.2f}，可能具有投注价值。"
            else:
                response += "根据概率和赔率计算，所有选项的期望价值都低于1.0，可能没有明显的投注价值。"
        
        return response


class ChatInterface:
    """聊天接口"""
    
    def __init__(self, config_path: str, model_path: str = None):
        """初始化聊天接口
        
        Args:
            config_path: 配置文件路径
            model_path: 模型路径
        """
        self.predictor = MatchPredictor(config_path, model_path)
        self.config = self.predictor.config
        
        # 系统提示
        self.system_prompt = self.config.get("chat", {}).get("system_prompt", 
            "你是一个专业的足球比赛预测专家，能够根据球队近期表现、赔率数据和专家分析来预测比赛结果。")
        
        # 对话历史
        self.history = []
        self.max_history = self.config.get("chat", {}).get("max_history", 10)
    
    def greet(self) -> str:
        """问候语
        
        Returns:
            问候语字符串
        """
        return f"你好！我是足球比赛预测专家。{self.system_prompt}\n\n请告诉我比赛信息，包括球队名称、近期状态和赔率等，我将为你预测比赛结果。"
    
    def respond(self, user_input: str) -> str:
        """响应用户输入
        
        Args:
            user_input: 用户输入
            
        Returns:
            响应字符串
        """
        # 添加到历史
        self.history.append({"role": "user", "content": user_input})
        
        # 检查是否是退出命令
        if user_input.lower() in ["退出", "exit", "quit", "再见"]:
            self.history.append({"role": "assistant", "content": "再见！祝您观赛愉快！"})
            return "再见！祝您观赛愉快！"
        
        # 检查是否是帮助命令
        if user_input.lower() in ["帮助", "help", "?"]:
            help_text = """
我可以帮助您预测足球比赛结果。请提供以下信息：
1. 比赛球队（主队和客队）
2. 球队近期状态（如：近5场比赛3胜1平1负）
3. 赔率信息（主胜、平局、客胜赔率）

示例输入：
"曼联主场对阵利物浦，曼联近3场2胜1负，利物浦近3场1胜2平，主胜2.1，平局3.4，客胜3.2"
            """
            self.history.append({"role": "assistant", "content": help_text})
            return help_text
        
        try:
            # 预测比赛
            prediction_result = self.predictor.predict_match(user_input)
            
            # 生成响应
            response = self.predictor.generate_response(prediction_result)
            
            # 添加到历史
            self.history.append({"role": "assistant", "content": response})
            
            # 限制历史长度
            if len(self.history) > self.max_history * 2:
                self.history = self.history[-self.max_history * 2:]
            
            return response
        except Exception as e:
            error_msg = f"抱歉，处理您的请求时出现错误：{str(e)}\n\n请检查输入信息是否完整，或尝试重新描述。"
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def run(self):
        """运行聊天界面"""
        print(self.greet())
        
        while True:
            try:
                user_input = input("\n您: ")
                
                if not user_input.strip():
                    continue
                
                response = self.respond(user_input)
                print(f"\n助手: {response}")
                
                # 检查是否退出
                if user_input.lower() in ["退出", "exit", "quit", "再见"]:
                    break
            except KeyboardInterrupt:
                print("\n\n再见！祝您观赛愉快！")
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="V5 Model Chat Interface")
    parser.add_argument("--config", type=str, default="configs/v5_config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    args = parser.parse_args()
    
    # 创建聊天接口
    chat_interface = ChatInterface(args.config, args.model)
    
    # 运行聊天界面
    chat_interface.run()


if __name__ == "__main__":
    main()