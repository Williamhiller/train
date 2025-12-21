import json
import os
from datetime import datetime
from collections import defaultdict

class TeamStateFeatureExtractor:
    def __init__(self, data_root):
        self.data_root = data_root

    def parse_match_date(self, date_str):
        """解析比赛日期，将"17-01-21"格式转换为datetime对象"""
        try:
            # 自动添加20作为世纪前缀
            return datetime.strptime(f"20{date_str}", "%Y-%m-%d")
        except ValueError:
            return None

    def extract_recent_form(self, match_data, team_id, is_home_team=True, num_matches=6):
        """提取球队近期战绩特征
        
        参数:
            match_data: 包含homeData和awayData的比赛数据
            team_id: 球队ID
            is_home_team: 是否为主队
            num_matches: 考虑的近期比赛数量
        
        返回:
            dict: 包含近期战绩特征的字典
        """
        # 确定使用主队还是客队的历史数据，注意数据存储在history字典中
        history = match_data.get('history', {})
        recent_data = history.get('homeData', []) if is_home_team else history.get('awayData', [])
        
        # 按日期排序，最近的比赛在前
        recent_data_sorted = sorted(recent_data, 
                                     key=lambda x: self.parse_match_date(x[0]) or datetime.min, 
                                     reverse=True)
        
        # 取最近的num_matches场比赛
        recent_matches = recent_data_sorted[:num_matches]
        
        # 计算战绩特征
        total_matches = len(recent_matches)
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for match in recent_matches:
            if len(match) >= 6:
                # 确定球队是主队还是客队
                is_team_home = match[2] == team_id
                
                # 获取进球数
                team_goals = match[4] if is_team_home else match[5]
                opponent_goals = match[5] if is_team_home else match[4]
                
                # 获取比赛结果（最后一个元素，索引为-1）
                result = match[-1]  # 3: 胜, 1: 平, 0: 负
                
                # 计算结果
                if result == 3:
                    wins += 1
                elif result == 1:
                    draws += 1
                else:
                    losses += 1
                
                goals_scored += team_goals
                goals_conceded += opponent_goals
        
        # 计算胜率、场均进球、场均失球等
        win_rate = wins / total_matches if total_matches > 0 else 0.0
        draw_rate = draws / total_matches if total_matches > 0 else 0.0
        loss_rate = losses / total_matches if total_matches > 0 else 0.0
        
        avg_goals_scored = goals_scored / total_matches if total_matches > 0 else 0.0
        avg_goals_conceded = goals_conceded / total_matches if total_matches > 0 else 0.0
        
        return {
            'recent_matches': total_matches,
            'recent_wins': wins,
            'recent_draws': draws,
            'recent_losses': losses,
            'win_rate': round(win_rate, 3),
            'draw_rate': round(draw_rate, 3),
            'loss_rate': round(loss_rate, 3),
            'avg_goals_scored': round(avg_goals_scored, 3),
            'avg_goals_conceded': round(avg_goals_conceded, 3)
        }

    def extract_season_data(self, match_data, is_home_team=True):
        """提取球队本赛季至今的数据特征
        
        参数:
            match_data: 包含homeSeasonData和awaySeasonData的比赛数据
            is_home_team: 是否为主队
        
        返回:
            dict: 包含本赛季数据特征的字典
        """
        history = match_data.get('history', {})
        
        if is_home_team:
            season_data = history.get('homeSeasonData', {})
        else:
            season_data = history.get('awaySeasonData', {})
        
        # 转换数据类型并处理可能的空值
        def safe_int(value, default=0):
            try:
                return int(value) if value and value != '-' else default
            except (ValueError, TypeError):
                return default
        
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str) and '%' in value:
                    return float(value.replace('%', ''))
                return float(value) if value and value != '-' else default
            except (ValueError, TypeError):
                return default
        
        return {
            'season_wins': safe_int(season_data.get('wins')),
            'season_draws': safe_int(season_data.get('draws')),
            'season_losses': safe_int(season_data.get('losses')),
            'season_goals_for': safe_int(season_data.get('goalsFor')),
            'season_goals_against': safe_int(season_data.get('goalsAgainst')),
            'season_points': safe_int(season_data.get('points')),
            'season_rank': safe_int(season_data.get('rank')),
            'season_win_rate': safe_float(season_data.get('winRate')) / 100.0  # 转换为0-1范围
        }

    def extract_head_to_head(self, match_data, home_team_id, away_team_id, num_matches=6):
        """提取两队对阵历史特征
        
        参数:
            match_data: 包含historyData的比赛数据
            home_team_id: 主队ID
            away_team_id: 客队ID
            num_matches: 考虑的对阵历史数量
        
        返回:
            dict: 包含对阵历史特征的字典
        """
        # 注意historyData存储在history字典中
        history_data = match_data.get('history', {}).get('historyData', [])
        
        # 筛选出两队之间的比赛
        head_to_head_matches = []
        for match in history_data:
            # 确保match有足够的元素
            if len(match) >= 6:
                # 正确的历史数据格式：
                # ["日期",主队ID,客队ID,?,主队进球数,客队进球数,比赛结果]
                # 例如：["25-02-06",28,19,2,0,0]
                match_home = match[1]
                match_away = match[2]
                
                # 检查是否是这两队之间的比赛
                if ((match_home == home_team_id and match_away == away_team_id) or
                    (match_home == away_team_id and match_away == home_team_id)):
                    head_to_head_matches.append(match)
        
        # 按日期排序，最近的比赛在前
        head_to_head_matches_sorted = sorted(head_to_head_matches, 
                                                key=lambda x: self.parse_match_date(x[0]) or datetime.min, 
                                                reverse=True)
        
        # 取最近的num_matches场比赛
        recent_head_to_head = head_to_head_matches_sorted[:num_matches]
        
        # 计算对阵历史特征
        total_matches = len(recent_head_to_head)
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        
        for match in recent_head_to_head:
            # 正确的历史数据格式：
            # ["日期",历史比赛主队ID,历史比赛客队ID,历史比赛主队进球数,历史比赛客队进球数,当前分析比赛主队在历史比赛中的结果]
            # 例如：["24-11-02",49,62,3,0,0] - 历史比赛49 vs 62，比分3-0，当前分析比赛主队62在这场比赛中失利
            match_home = match[1]
            match_away = match[2]
            match_home_goals = match[3]
            match_away_goals = match[4]
            result = match[-1]  # 0: 当前分析比赛主队在历史比赛中失利, 3: 当前分析比赛主队在历史比赛中获胜
            
            # 计算当前分析比赛主队和客队的进球数
            # 当前分析比赛主队是home_team_id，客队是away_team_id
            if match_home == home_team_id and match_away == away_team_id:
                # 历史比赛中当前分析比赛主队是主队
                home_goals += match_home_goals
                away_goals += match_away_goals
            elif match_home == away_team_id and match_away == home_team_id:
                # 历史比赛中当前分析比赛主队是客队
                home_goals += match_away_goals
                away_goals += match_home_goals
            
            # 根据结果计算胜平负
            if result == 3:
                # 当前分析比赛主队在历史比赛中获胜
                home_wins += 1
            elif result == 1:
                # 平局
                draws += 1
            else:  # 0或其他值
                # 当前分析比赛主队在历史比赛中失利
                away_wins += 1
        
        # 计算胜率
        home_win_rate = home_wins / total_matches if total_matches > 0 else 0.0
        away_win_rate = away_wins / total_matches if total_matches > 0 else 0.0
        draw_rate = draws / total_matches if total_matches > 0 else 0.0
        
        return {
            'head_to_head_matches': total_matches,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'draw_rate': draw_rate
        }

    def extract_match_team_state_features(self, match_data, home_team_id, away_team_id):
        """提取一场比赛的球队状态特征
        
        参数:
            match_data: 包含history数据的比赛数据
            home_team_id: 主队ID
            away_team_id: 客队ID
        
        返回:
            dict: 包含所有球队状态特征的字典
        """
        # 提取主队近期战绩
        home_recent_form = self.extract_recent_form(match_data, home_team_id, is_home_team=True)
        
        # 提取主队本赛季数据
        home_season_data = self.extract_season_data(match_data, is_home_team=True)
        
        # 提取客队近期战绩
        away_recent_form = self.extract_recent_form(match_data, away_team_id, is_home_team=False)
        
        # 提取客队本赛季数据
        away_season_data = self.extract_season_data(match_data, is_home_team=False)
        
        # 提取对阵历史
        head_to_head = self.extract_head_to_head(match_data, home_team_id, away_team_id)
        
        # 合并所有特征
        features = {
            'home_team': {
                'id': home_team_id,
                'recent_form': home_recent_form,
                'season_data': home_season_data
            },
            'away_team': {
                'id': away_team_id,
                'recent_form': away_recent_form,
                'season_data': away_season_data
            },
            'head_to_head': head_to_head
        }
        
        return features

    def extract_season_team_state_features(self, season):
        """提取整个赛季的球队状态特征
        
        参数:
            season: 赛季名称（如"2017-2018"）
        
        返回:
            dict: 包含整个赛季所有比赛的球队状态特征的字典，包括赛果信息
        """
        season_dir = os.path.join(self.data_root, season)
        details_dir = os.path.join(season_dir, "details")
        
        if not os.path.exists(details_dir):
            print(f"赛季 {season} 的数据目录不存在: {details_dir}")
            return None
        
        # 读取赛季比赛列表，获取主队和客队ID以及赛果信息
        season_match_list_path = os.path.join(season_dir, f"36_{season}.json")
        match_id_to_info = {}
        
        if os.path.exists(season_match_list_path):
            with open(season_match_list_path, 'r', encoding='utf-8') as f:
                season_match_data = json.load(f)
                # 赛季比赛列表是字典结构，直接遍历所有值
                for match in season_match_data.values():
                    match_id = str(match.get('matchId'))
                    home_team_id = match.get('homeTeamId')
                    away_team_id = match.get('awayTeamId')
                    result = match.get('result')  # 3: 主队胜, 1: 平, 0: 客队胜
                    if match_id and home_team_id and away_team_id:
                        match_id_to_info[match_id] = {
                            'home': home_team_id,
                            'away': away_team_id,
                            'result': result
                        }
        
        season_features = {}
        
        # 遍历details目录下的所有子目录和文件
        for root, dirs, files in os.walk(details_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        # 读取比赛数据
                        with open(file_path, 'r', encoding='utf-8') as f:
                            match_data = json.load(f)
                        
                        # 获取比赛ID
                        match_id = os.path.splitext(file_name)[0]
                        
                        # 获取主队和客队ID以及赛果
                        match_info = match_id_to_info.get(match_id)
                        if not match_info:
                            print(f"比赛 {match_id} 缺少主队或客队ID")
                            continue
                        
                        home_team_id = match_info['home']
                        away_team_id = match_info['away']
                        result = match_info['result']
                        
                        # 提取球队状态特征
                        team_state_features = self.extract_match_team_state_features(match_data, home_team_id, away_team_id)
                        
                        if team_state_features:
                            # 添加赛果信息到特征中
                            team_state_features['result'] = result
                            team_state_features['result_description'] = {
                                3: '主队胜',
                                1: '平局',
                                0: '客队胜'
                            }.get(result, '未知')
                            
                            season_features[match_id] = team_state_features
                            print(f"已提取比赛 {match_id} 的球队状态特征")
                        else:
                            print(f"比赛 {match_id} 缺少足够的历史数据来提取特征")
                            
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        return season_features

    def save_features(self, features, output_path):
        """保存特征到JSON文件
        
        参数:
            features: 要保存的特征数据
            output_path: 输出文件路径
        """
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存特征到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    data_root = "/Users/Williamhiler/Documents/my-project/train/original-data"
    train_data_root = "/Users/Williamhiler/Documents/my-project/train/train-data"
    extractor = TeamStateFeatureExtractor(data_root)
    
    # 获取所有赛季目录
    season_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for season in season_dirs:
        print(f"正在处理赛季 {season} 的球队状态特征...")
        season_features = extractor.extract_season_team_state_features(season)
        
        if season_features:
            output_path = os.path.join(train_data_root, "team_state", f"{season}_team_state_features.json")
            extractor.save_features(season_features, output_path)
            print(f"赛季 {season} 的球队状态特征已保存到 {output_path}")
        else:
            print(f"未找到赛季 {season} 的数据")
    
    print("所有赛季球队状态特征提取完成！")