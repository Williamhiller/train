import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
from datetime import datetime, timedelta

from .context_extractor import ContextExtractor


class MatchDataProcessor:
    """比赛数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.label_column = "result"
        # 初始化上下文提取器
        self.context_extractor = ContextExtractor()
        self.initialized = False
        
    def load_raw_data(self, data_path: str) -> pd.DataFrame:
        """加载原始数据
        
        Args:
            data_path: 数据路径
            
        Returns:
            原始数据DataFrame
        """
        # 获取历史数据配置
        historical_config = self.config.get("data", {}).get("historical_data", {})
        use_historical = historical_config.get("use_historical", True)
        seasons_to_load = historical_config.get("seasons", ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"])
        
        # 加载比赛数据
        all_data = []
        
        # 遍历所有赛季数据
        for season_dir in os.listdir(data_path):
            if not season_dir.startswith('20'):
                continue
                
            # 检查是否需要加载该赛季
            if use_historical and season_dir not in seasons_to_load:
                continue
                
            season_path = os.path.join(data_path, season_dir)
            if not os.path.isdir(season_path):
                continue
                
            # 加载赛季概览数据
            overview_file = os.path.join(season_path, f"36_{season_dir}.json")
            if os.path.exists(overview_file):
                with open(overview_file, 'r', encoding='utf-8') as f:
                    season_data = json.load(f)
                    # 检查数据类型，如果是字典则转换为列表
                    if isinstance(season_data, dict):
                        # 获取字典的值列表
                        season_data_list = list(season_data.values())
                    elif isinstance(season_data, list):
                        season_data_list = season_data
                    else:
                        season_data_list = []
                    # 添加赛季信息
                    for match in season_data_list:
                        if isinstance(match, dict):
                            match["season"] = season_dir
                    all_data.extend(season_data_list)
        
        return pd.DataFrame(all_data)
    
    def load_match_details(self, data_path: str, match_ids: List[str]) -> Dict[str, Dict]:
        """加载比赛详情
        
        Args:
            data_path: 数据路径
            match_ids: 比赛ID列表
            
        Returns:
            比赛详情字典
        """
        match_details = {}
        
        for season_dir in os.listdir(data_path):
            if not season_dir.startswith('20'):
                continue
                
            season_path = os.path.join(data_path, season_dir)
            details_path = os.path.join(season_path, "details")
            
            if not os.path.isdir(details_path):
                continue
                
            # 遍历所有轮次
            for round_dir in os.listdir(details_path):
                round_path = os.path.join(details_path, round_dir)
                if not os.path.isdir(round_path):
                    continue
                    
                # 加载该轮次的所有比赛
                for match_file in os.listdir(round_path):
                    if not match_file.endswith('.json'):
                        continue
                        
                    match_id = match_file.replace('.json', '')
                    if match_id in match_ids:
                        file_path = os.path.join(round_path, match_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            match_details[match_id] = json.load(f)
        
        return match_details
    
    def extract_team_form(self, team_id: str, matches: List[Dict], current_date: str, num_matches: int = 5) -> Dict:
        """提取球队近期状态
        
        Args:
            team_id: 球队ID
            matches: 比赛列表
            current_date: 当前日期
            num_matches: 考虑的最近比赛数量
            
        Returns:
            球队近期状态字典
        """
        # 筛选该球队的比赛
        team_matches = []
        for match in matches:
            if match.get("homeTeamId") == team_id or match.get("awayTeamId") == team_id:
                team_matches.append(match)
        
        # 按日期排序
        team_matches.sort(key=lambda x: x.get("matchTime", ""))
        
        # 获取当前日期之前的比赛
        past_matches = []
        for match in team_matches:
            if match.get("matchTime", "") < current_date:
                past_matches.append(match)
        
        # 取最近的num_matches场比赛
        recent_matches = past_matches[-num_matches:] if len(past_matches) >= num_matches else past_matches
        
        if not recent_matches:
            return {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_scored": 0,
                "goals_conceded": 0,
                "goal_difference": 0,
                "form_points": 0,
                "form_string": "",
                "matches_played": 0
            }
        
        # 统计数据
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        form_string = ""
        
        for match in recent_matches:
            is_home = match.get("homeTeamId") == team_id
            
            if is_home:
                team_score = match.get("homeScore", 0)
                opponent_score = match.get("awayScore", 0)
            else:
                team_score = match.get("awayScore", 0)
                opponent_score = match.get("homeScore", 0)
            
            goals_scored += team_score
            goals_conceded += opponent_score
            
            if team_score > opponent_score:
                wins += 1
                form_string = "W" + form_string
            elif team_score == opponent_score:
                draws += 1
                form_string = "D" + form_string
            else:
                losses += 1
                form_string = "L" + form_string
        
        # 计算状态积分
        form_points = wins * 3 + draws
        
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "form_points": form_points,
            "form_string": form_string,
            "matches_played": len(recent_matches)
        }
    
    def extract_odds_features(self, match_odds: Dict) -> Dict:
        """提取赔率特征
        
        Args:
            match_odds: 比赛赔率数据
            
        Returns:
            赔率特征字典
        """
        if not match_odds:
            return {
                "home_win_odds": 0.0,
                "draw_odds": 0.0,
                "away_win_odds": 0.0,
                "over_under_odds": 0.0,
                "asian_handicap_odds": 0.0,
                "implied_home_win": 0.0,
                "implied_draw": 0.0,
                "implied_away_win": 0.0
            }
        
        # 菠菜公司ID映射
        bookmaker_map = {
            "82": "william",  # 威廉
            "115": "ladbrokes"  # 立博
        }
        
        # 基础赔率
        home_win_odds = match_odds.get("home_win", 0.0)
        draw_odds = match_odds.get("draw", 0.0)
        away_win_odds = match_odds.get("away_win", 0.0)
        
        # 计算隐含概率
        total_implied = 0.0
        if home_win_odds > 0:
            total_implied += 1.0 / home_win_odds
        if draw_odds > 0:
            total_implied += 1.0 / draw_odds
        if away_win_odds > 0:
            total_implied += 1.0 / away_win_odds
        
        implied_home_win = (1.0 / home_win_odds) / total_implied if home_win_odds > 0 else 0.0
        implied_draw = (1.0 / draw_odds) / total_implied if draw_odds > 0 else 0.0
        implied_away_win = (1.0 / away_win_odds) / total_implied if away_win_odds > 0 else 0.0
        
        # 创建赔率特征字典
        odds_features = {
            "home_win_odds": home_win_odds,
            "draw_odds": draw_odds,
            "away_win_odds": away_win_odds,
            "over_under_odds": match_odds.get("over_under", 0.0),
            "asian_handicap_odds": match_odds.get("asian_handicap", 0.0),
            "implied_home_win": implied_home_win,
            "implied_draw": implied_draw,
            "implied_away_win": implied_away_win
        }
        
        # 处理详细的菠菜公司赔率
        for company_id, odds_data in match_odds.items():
            # 跳过oddId和其他非公司ID字段
            if company_id in ["oddId"]:
                continue
                
            # 检查是否为我们关注的菠菜公司
            if company_id in bookmaker_map and isinstance(odds_data, list) and len(odds_data) > 0:
                bookmaker = bookmaker_map[company_id]
                
                # 取最新的赔率记录（列表第一个元素）
                latest_odds = odds_data[0]
                if len(latest_odds) >= 3:
                    # 提取主胜、平局、客胜赔率
                    home_odds = float(latest_odds[0])
                    draw_odds = float(latest_odds[1])
                    away_odds = float(latest_odds[2])
                    
                    # 添加到赔率特征字典
                    odds_features[f"{bookmaker}_home_odds"] = home_odds
                    odds_features[f"{bookmaker}_draw_odds"] = draw_odds
                    odds_features[f"{bookmaker}_away_odds"] = away_odds
                    
                    # 计算隐含概率
                    total = 0.0
                    if home_odds > 0:
                        total += 1.0 / home_odds
                    if draw_odds > 0:
                        total += 1.0 / draw_odds
                    if away_odds > 0:
                        total += 1.0 / away_odds
                    
                    if total > 0:
                        odds_features[f"{bookmaker}_implied_home"] = (1.0 / home_odds) / total if home_odds > 0 else 0.0
                        odds_features[f"{bookmaker}_implied_draw"] = (1.0 / draw_odds) / total if draw_odds > 0 else 0.0
                        odds_features[f"{bookmaker}_implied_away"] = (1.0 / away_odds) / total if away_odds > 0 else 0.0
        
        return odds_features
    
    def extract_head_to_head(self, home_team_id: str, away_team_id: str, all_matches: List[Dict], match_date: str, num_matches: int = 10) -> Dict:
        """提取两队交锋历史
        
        Args:
            home_team_id: 主队ID
            away_team_id: 客队ID
            all_matches: 所有比赛数据
            match_date: 当前日期
            num_matches: 考虑的最近比赛数量
            
        Returns:
            交锋历史字典
        """
        # 筛选两队之间的比赛
        h2h_matches = []
        for match in all_matches:
            home = match.get("homeTeamId")
            away = match.get("awayTeamId")
            date = match.get("matchTime", "")
            
            # 检查是否是两队之间的比赛且在当前日期之前
            if date < match_date and ((home == home_team_id and away == away_team_id) or (home == away_team_id and away == home_team_id)):
                h2h_matches.append(match)
        
        # 按日期排序
        h2h_matches.sort(key=lambda x: x.get("date", ""))
        
        # 取最近的num_matches场比赛
        recent_h2h = h2h_matches[-num_matches:] if len(h2h_matches) >= num_matches else h2h_matches
        
        if not recent_h2h:
            return {
                "total_matches": 0,
                "home_wins": 0,
                "away_wins": 0,
                "draws": 0,
                "home_goals": 0,
                "away_goals": 0,
                "home_win_rate": 0.0,
                "away_win_rate": 0.0,
                "draw_rate": 0.0
            }
        
        # 统计数据
        home_wins = away_wins = draws = 0
        home_goals = away_goals = 0
        
        for match in recent_h2h:
            home = match.get("homeTeamId")
            away = match.get("awayTeamId")
            home_score = match.get("homeScore", 0)
            away_score = match.get("awayScore", 0)
            
            home_goals += home_score
            away_goals += away_score
            
            if home_score > away_score:
                if home == home_team_id:
                    home_wins += 1
                else:
                    away_wins += 1
            elif home_score < away_score:
                if away == home_team_id:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                draws += 1
        
        total_matches = len(recent_h2h)
        
        return {
            "total_matches": total_matches,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_win_rate": home_wins / total_matches,
            "away_win_rate": away_wins / total_matches,
            "draw_rate": draws / total_matches
        }
    
    def extract_historical_performance(self, team_id: str, all_matches: List[Dict], match_date: str) -> Dict:
        """提取球队历史表现
        
        Args:
            team_id: 球队ID
            all_matches: 所有比赛数据
            match_date: 当前日期
            
        Returns:
            历史表现字典
        """
        # 筛选该球队的所有比赛
        team_matches = []
        for match in all_matches:
            if match.get("homeTeamId") == team_id or match.get("awayTeamId") == team_id:
                team_matches.append(match)
        
        # 按日期排序
        team_matches.sort(key=lambda x: x.get("matchTime", ""))
        
        # 分割为历史数据和当前赛季数据
        historical_matches = []
        current_season_matches = []
        
        for match in team_matches:
            if match.get("matchTime", "") < match_date:
                # 检查是否是当前赛季
                match_season = match.get("season", "")
                if match_season:
                    historical_matches.append(match)
                else:
                    historical_matches.append(match)
        
        # 统计历史表现
        if not historical_matches:
            return {
                "total_matches": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "goals_per_match": 0.0,
                "goals_conceded_per_match": 0.0,
                "points_per_match": 0.0
            }
        
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        
        for match in historical_matches:
            is_home = match.get("homeTeamId") == team_id
            
            if is_home:
                team_score = match.get("homeScore", 0)
                opponent_score = match.get("awayScore", 0)
            else:
                team_score = match.get("awayScore", 0)
                opponent_score = match.get("homeScore", 0)
            
            goals_scored += team_score
            goals_conceded += opponent_score
            
            if team_score > opponent_score:
                wins += 1
            elif team_score == opponent_score:
                draws += 1
            else:
                losses += 1
        
        total_matches = len(historical_matches)
        total_points = wins * 3 + draws
        
        return {
            "total_matches": total_matches,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / total_matches,
            "draw_rate": draws / total_matches,
            "loss_rate": losses / total_matches,
            "goals_per_match": goals_scored / total_matches,
            "goals_conceded_per_match": goals_conceded / total_matches,
            "points_per_match": total_points / total_matches
        }
    
    def create_match_features(self, match: Dict, all_matches: List[Dict], match_details: Dict) -> Dict:
        """创建比赛特征
        
        Args:
            match: 比赛数据
            all_matches: 所有比赛数据
            match_details: 比赛详情
            
        Returns:
            比赛特征字典
        """
        match_id = str(match.get("matchId", ""))
        home_team_id = match.get("homeTeamId", "")
        away_team_id = match.get("awayTeamId", "")
        match_date = match.get("matchTime", "")
        
        # 获取球队近期状态
        home_form = self.extract_team_form(home_team_id, all_matches, match_date)
        away_form = self.extract_team_form(away_team_id, all_matches, match_date)
        
        # 获取球队历史表现
        home_historical = self.extract_historical_performance(home_team_id, all_matches, match_date)
        away_historical = self.extract_historical_performance(away_team_id, all_matches, match_date)
        
        # 获取交锋历史
        h2h_history = self.extract_head_to_head(home_team_id, away_team_id, all_matches, match_date)
        
        # 获取比赛详情
        detail = match_details.get(match_id, {})
        
        # 获取赔率特征 - 从detail中提取，因为赛季文件中的odds是空的
        odds_features = self.extract_odds_features(detail.get("odds", {}))
        
        # 从比赛详情中提取更完整的信息
        # 提取球队名称
        home_team_name = detail.get("homeTeam", detail.get("homeTeamName", match.get("homeTeamName", home_team_id)))
        away_team_name = detail.get("awayTeam", detail.get("awayTeamName", match.get("awayTeamName", away_team_id)))
        
        # 获取比赛详情中的history字段
        history = detail.get("history", {})
        
        # 提取比分
        home_score = detail.get("homeScore", detail.get("home_score", 0))
        away_score = detail.get("awayScore", detail.get("away_score", 0))
        
        # 创建特征字典
        features = {
            # 比赛基本信息
            "matchId": match_id,
            "match_id": match_id,
            "homeTeamId": home_team_id,
            "awayTeamId": away_team_id,
            "homeTeam": home_team_name,
            "homeTeamName": home_team_name,
            "awayTeam": away_team_name,
            "awayTeamName": away_team_name,
            "match_date": match_date,
            "season": match.get("season", ""),
            
            # 主队近期状态
            "home_wins": home_form["wins"],
            "home_draws": home_form["draws"],
            "home_losses": home_form["losses"],
            "home_goals_scored": home_form["goals_scored"],
            "home_goals_conceded": home_form["goals_conceded"],
            "home_goal_difference": home_form["goal_difference"],
            "home_form_points": home_form["form_points"],
            "home_matches_played": home_form["matches_played"],
            "home_form_string": home_form["form_string"],
            
            # 客队近期状态
            "away_wins": away_form["wins"],
            "away_draws": away_form["draws"],
            "away_losses": away_form["losses"],
            "away_goals_scored": away_form["goals_scored"],
            "away_goals_conceded": away_form["goals_conceded"],
            "away_goal_difference": away_form["goal_difference"],
            "away_form_points": away_form["form_points"],
            "away_matches_played": away_form["matches_played"],
            "away_form_string": away_form["form_string"],
            
            # 主队本赛季数据
            "home_rank": "",  # 可以从赛季数据中提取，这里先留空
            "home_win_rate": home_historical["win_rate"] * 100,  # 转换为百分比
            "home_goals_for": int(home_historical["goals_per_match"] * home_historical["total_matches"]),
            "home_goals_against": int(home_historical["goals_conceded_per_match"] * home_historical["total_matches"]),
            
            # 客队本赛季数据
            "away_rank": "",  # 可以从赛季数据中提取，这里先留空
            "away_win_rate": away_historical["win_rate"] * 100,  # 转换为百分比
            "away_goals_for": int(away_historical["goals_per_match"] * away_historical["total_matches"]),
            "away_goals_against": int(away_historical["goals_conceded_per_match"] * away_historical["total_matches"]),
            
            # 主队历史表现
            "home_historical_wins": home_historical["wins"],
            "home_historical_draws": home_historical["draws"],
            "home_historical_losses": home_historical["losses"],
            "home_historical_win_rate": home_historical["win_rate"],
            "home_historical_goals_per_match": home_historical["goals_per_match"],
            "home_historical_goals_conceded_per_match": home_historical["goals_conceded_per_match"],
            "home_historical_points_per_match": home_historical["points_per_match"],
            
            # 客队历史表现
            "away_historical_wins": away_historical["wins"],
            "away_historical_draws": away_historical["draws"],
            "away_historical_losses": away_historical["losses"],
            "away_historical_win_rate": away_historical["win_rate"],
            "away_historical_goals_per_match": away_historical["goals_per_match"],
            "away_historical_goals_conceded_per_match": away_historical["goals_conceded_per_match"],
            "away_historical_points_per_match": away_historical["points_per_match"],
            
            # 交锋历史
            "h2h_total_matches": h2h_history["total_matches"],
            "h2h_home_wins": h2h_history["home_wins"],
            "h2h_away_wins": h2h_history["away_wins"],
            "h2h_draws": h2h_history["draws"],
            "h2h_home_win_rate": h2h_history["home_win_rate"],
            "h2h_away_win_rate": h2h_history["away_win_rate"],
            "h2h_draw_rate": h2h_history["draw_rate"],
            "head_to_head_history": f"两队交锋{h2h_history['total_matches']}场，{home_team_name}赢{h2h_history['home_wins']}场，{away_team_name}赢{h2h_history['away_wins']}场，平局{h2h_history['draws']}场",
            
            # 赔率特征 - 基础赔率
            "home_win_odds": odds_features["home_win_odds"],
            "draw_odds": odds_features["draw_odds"],
            "away_win_odds": odds_features["away_win_odds"],
            "over_under_odds": odds_features["over_under_odds"],
            "asian_handicap_odds": odds_features["asian_handicap_odds"],
            "implied_home_win": odds_features["implied_home_win"],
            "implied_draw": odds_features["implied_draw"],
            "implied_away_win": odds_features["implied_away_win"],
            
            # 威廉希尔赔率特征
            "william_home_odds": odds_features.get("william_home_odds", 0.0),
            "william_draw_odds": odds_features.get("william_draw_odds", 0.0),
            "william_away_odds": odds_features.get("william_away_odds", 0.0),
            "william_implied_home": odds_features.get("william_implied_home", 0.0),
            "william_implied_draw": odds_features.get("william_implied_draw", 0.0),
            "william_implied_away": odds_features.get("william_implied_away", 0.0),
            
            # 立博赔率特征
            "ladbrokes_home_odds": odds_features.get("ladbrokes_home_odds", 0.0),
            "ladbrokes_draw_odds": odds_features.get("ladbrokes_draw_odds", 0.0),
            "ladbrokes_away_odds": odds_features.get("ladbrokes_away_odds", 0.0),
            "ladbrokes_implied_home": odds_features.get("ladbrokes_implied_home", 0.0),
            "ladbrokes_implied_draw": odds_features.get("ladbrokes_implied_draw", 0.0),
            "ladbrokes_implied_away": odds_features.get("ladbrokes_implied_away", 0.0),
            
            # 原始赔率数据 - 用于context_generator计算初赔和终赔
            "odds": match.get("odds", {}),
            
            # 比赛结果（标签）
            "result": 0,
            "homeScore": home_score,
            "home_score": home_score,
            "awayScore": away_score,
            "away_score": away_score,
            
            # 从比赛详情中获取的history字段，用于context_generator
            "history": history,
            
            # 文本描述特征
            "historical_performance": f"{home_team_name}历史表现：{home_historical['win_rate']:.1%}胜率，{home_historical['points_per_match']:.1f}分/场；{away_team_name}历史表现：{away_historical['win_rate']:.1%}胜率，{away_historical['points_per_match']:.1f}分/场"
        }
        
        # 计算比赛结果
        home_score = features["home_score"]
        away_score = features["away_score"]
        
        if home_score > away_score:
            features["result"] = 0  # 主队赢
        elif home_score == away_score:
            features["result"] = 1  # 平局
        else:
            features["result"] = 2  # 客队赢
        
        # 生成用于知识匹配的上下文
        knowledge_matching_context = self.context_extractor.extract_context(
            match_data=features, include_result=False, context_type='knowledge_matching'
        )
        features['knowledge_matching_context'] = knowledge_matching_context
        
        # 生成用于预测的上下文
        prediction_context = self.context_extractor.extract_context(
            match_data=features, include_result=False, context_type='prediction'
        )
        features['prediction_context'] = prediction_context
        
        return features
    
    def process_data(self, data_path: str, batch_size: Optional[int] = None, save_by_season: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """处理比赛数据，支持批次处理和按赛季保存
        
        Args:
            data_path: 数据路径
            batch_size: 批次大小，为None时处理所有数据
            save_by_season: 是否按赛季保存数据
            
        Returns:
            如果save_by_season为True，返回赛季->DataFrame的字典；否则返回单个DataFrame
        """
        # 加载原始数据
        raw_data = self.load_raw_data(data_path)
        
        # 如果指定了批次大小，只处理指定批次的数据
        if batch_size:
            print(f"Processing batch of {batch_size} matches...")
            raw_data = raw_data.head(batch_size)
        
        # 按赛季分组处理
        if save_by_season and 'season' in raw_data.columns:
            season_dfs = {}
            for season in raw_data['season'].unique():
                print(f"Processing season: {season}")
                season_data = raw_data[raw_data['season'] == season]
                
                # 获取该赛季的比赛ID
                match_ids = season_data["matchId"].astype(str).tolist()
                
                # 加载该赛季的比赛详情
                match_details = self.load_match_details(data_path, match_ids)
                
                # 创建特征
                all_features = []
                for _, match in season_data.iterrows():
                    match_dict = match.to_dict()
                    features = self.create_match_features(match_dict, season_data.to_dict("records"), match_details)
                    all_features.append(features)
                
                # 转换为DataFrame
                features_df = pd.DataFrame(all_features)
                
                # 跳过空数据框
                if len(features_df) == 0:
                    print(f"Season {season}: No data available, skipping...")
                    continue
                
                # 填充缺失值
                features_df[self.feature_columns] = features_df[self.feature_columns].fillna(0)
                
                # 保存该赛季的数据
                season_dfs[season] = features_df
                
                print(f"Season {season}: {len(features_df)} matches processed")
            
            return season_dfs
        else:
            # 原有逻辑：处理所有数据
            # 获取所有比赛ID
            match_ids = raw_data["matchId"].astype(str).tolist()
            
            # 加载比赛详情
            match_details = self.load_match_details(data_path, match_ids)
            
            # 创建特征
            all_features = []
            for _, match in raw_data.iterrows():
                match_dict = match.to_dict()
                features = self.create_match_features(match_dict, raw_data.to_dict("records"), match_details)
                all_features.append(features)
            
            # 转换为DataFrame
            features_df = pd.DataFrame(all_features)
            
            # 定义特征列
            self.feature_columns = [
                # 主队近期状态
                "home_wins", "home_draws", "home_losses", "home_goals_scored", "home_goals_conceded",
                "home_goal_difference", "home_form_points", "home_matches_played",
                
                # 客队近期状态
                "away_wins", "away_draws", "away_losses", "away_goals_scored", "away_goals_conceded",
                "away_goal_difference", "away_form_points", "away_matches_played",
                
                # 主队历史表现
                "home_historical_wins", "home_historical_draws", "home_historical_losses", 
                "home_historical_win_rate", "home_historical_goals_per_match", "home_historical_points_per_match",
                
                # 客队历史表现
                "away_historical_wins", "away_historical_draws", "away_historical_losses", 
                "away_historical_win_rate", "away_historical_goals_per_match", "away_historical_points_per_match",
                
                # 交锋历史
                "h2h_total_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
                "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
                
                # 赔率特征
                "home_win_odds", "draw_odds", "away_win_odds", "over_under_odds", "asian_handicap_odds",
                "implied_home_win", "implied_draw", "implied_away_win"
            ]
            
            # 填充缺失值
            features_df[self.feature_columns] = features_df[self.feature_columns].fillna(0)
            
            # 标准化特征
            features_df[self.feature_columns] = self.scaler.fit_transform(features_df[self.feature_columns])
            
            return features_df
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名
        
        Returns:
            特征列名列表
        """
        return self.feature_columns
    
    def initialize_scaler(self, all_data: pd.DataFrame):
        """初始化scaler，使用所有数据"""
        if not self.initialized:
            print("Initializing scaler with all data...")
            self.scaler.fit(all_data[self.feature_columns])
            self.initialized = True
    
    def get_label_column(self) -> str:
        """获取标签列名
        
        Returns:
            标签列名
        """
        return self.label_column