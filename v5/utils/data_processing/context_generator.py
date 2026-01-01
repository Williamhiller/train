import json
from typing import Dict, List, Optional, Any

class ContextGenerator:
    """上下文生成器，结合专家知识库和原始数据生成更丰富的上下文"""
    
    def __init__(self):
        self.context_dimensions = [
            'basic_info',
            'odds_info', 
            'team_performance',
            'recent_form',
            'head_to_head',
            'season_data'
        ]
        
        # 菠菜公司ID与名称映射
        self.bookmaker_map = {
            '82': '威廉',
            '115': '立博'
        }
    
    def load_match_data(self, file_path: str) -> Dict[str, Any]:
        """加载比赛数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_initial_odds(self, odds_data: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        """获取初赔数据（最早的赔率记录）"""
        initial_odds = {}
        for company_id, odds_list in odds_data.items():
            if isinstance(odds_list, list) and len(odds_list) > 0:
                # 赔率列表按时间倒序排列，取最后一个作为初赔
                initial_odds[company_id] = odds_list[-1]
        return initial_odds
    
    def get_final_odds(self, odds_data: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        """获取终赔数据（最新的赔率记录）"""
        final_odds = {}
        for company_id, odds_list in odds_data.items():
            if isinstance(odds_list, list) and len(odds_list) > 0:
                # 赔率列表按时间倒序排列，取第一个作为终赔
                final_odds[company_id] = odds_list[0]
        return final_odds
    
    def analyze_odds_change(self, initial_odds: Dict[str, List[str]], final_odds: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """分析赔率变化"""
        odds_change = {}
        
        for company_id in initial_odds:
            if company_id in final_odds:
                initial = initial_odds[company_id]
                final = final_odds[company_id]
                
                # 提取胜平负赔率并计算变化
                home_initial = float(initial[0])
                draw_initial = float(initial[1])
                away_initial = float(initial[2])
                
                home_final = float(final[0])
                draw_final = float(final[1])
                away_final = float(final[2])
                
                odds_change[company_id] = {
                    'home_change': home_final - home_initial,
                    'draw_change': draw_final - draw_initial,
                    'away_change': away_final - away_initial,
                    'home_percent_change': ((home_final - home_initial) / home_initial) * 100 if home_initial != 0 else 0,
                    'draw_percent_change': ((draw_final - draw_initial) / draw_initial) * 100 if draw_initial != 0 else 0,
                    'away_percent_change': ((away_final - away_initial) / away_initial) * 100 if away_initial != 0 else 0
                }
        
        return odds_change
    
    def calculate_recent_form(self, match_history: List[List[str]], num_matches: int = 5) -> Dict[str, float]:
        """计算近期状态"""
        # 取最近n场比赛
        recent_matches = match_history[:num_matches]
        total_matches = len(recent_matches)
        
        if total_matches == 0:
            return {
                'win_rate': 0.0,
                'draw_rate': 0.0,
                'loss_rate': 0.0,
                'goals_per_match': 0.0,
                'goals_against_per_match': 0.0
            }
        
        total_wins = 0
        total_draws = 0
        total_goals = 0
        total_goals_against = 0
        
        for match in recent_matches:
            # 历史数据格式：[date, home_id, away_id, home_goals, away_goals, ?]
            # 对于homeData：match[3]是进球，match[4]是失球
            # 对于awayData：match[4]是进球，match[3]是失球
            # 这里简化处理，假设match[3]是己方进球，match[4]是对方进球
            home_goals = int(match[3])
            away_goals = int(match[4])
            
            total_goals += home_goals
            total_goals_against += away_goals
            
            if home_goals > away_goals:
                total_wins += 1
            elif home_goals == away_goals:
                total_draws += 1
        
        total_losses = total_matches - total_wins - total_draws
        
        return {
            'win_rate': total_wins / total_matches,
            'draw_rate': total_draws / total_matches,
            'loss_rate': total_losses / total_matches,
            'goals_per_match': total_goals / total_matches,
            'goals_against_per_match': total_goals_against / total_matches
        }
    
    def calculate_last_n_matches(self, match_history: List[List[str]], num_matches: int = 6) -> Dict[str, int]:
        """计算近n场比赛的胜平负数量"""
        # 取最近n场比赛
        recent_matches = match_history[:num_matches]
        
        wins = 0
        draws = 0
        losses = 0
        
        for match in recent_matches:
            # 历史数据格式：[date, home_id, away_id, home_goals, away_goals, ?]
            # 对于homeData：match[3]是进球，match[4]是失球
            # 对于awayData：match[4]是进球，match[3]是失球
            # 这里简化处理，假设match[3]是己方进球，match[4]是对方进球
            home_goals = int(match[3])
            away_goals = int(match[4])
            
            if home_goals > away_goals:
                wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        
        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'total': len(recent_matches)
        }
    
    def analyze_head_to_head(self, history_data: List[List[str]]) -> Dict[str, int]:
        """分析交锋历史"""
        home_wins = 0
        away_wins = 0
        draws = 0
        
        for match in history_data:
            home_goals = int(match[3])
            away_goals = int(match[4])
            
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1
        
        total_matches = home_wins + away_wins + draws
        
        return {
            'total_matches': total_matches,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws,
            'home_win_rate': home_wins / total_matches if total_matches > 0 else 0,
            'away_win_rate': away_wins / total_matches if total_matches > 0 else 0,
            'draw_rate': draws / total_matches if total_matches > 0 else 0
        }
    
    def _generate_team_status(self, recent_wins: int, recent_draws: int, recent_losses: int) -> str:
        """生成球队近期状态字符串，如'WDLWD'"""
        # 确保输入值为非负整数
        recent_wins = max(0, int(recent_wins))
        recent_draws = max(0, int(recent_draws))
        recent_losses = max(0, int(recent_losses))
        
        # 生成状态字符串，W=胜，D=平，L=负
        status_str = ''
        status_str += 'W' * recent_wins
        status_str += 'D' * recent_draws
        status_str += 'L' * recent_losses
        return status_str
    
    def generate_context_for_knowledge_matching(self, match_data: Dict[str, Any], include_result: bool = False) -> str:
        """生成用于知识匹配的上下文
        
        Args:
            match_data: 比赛数据
            include_result: 是否包含赛果信息
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        # 1. 比赛基本信息
        match_id = match_data.get('matchId', match_data.get('match_id', '未知'))
        context_parts.append(f"比赛ID: {match_id}")
        
        # 2. 球队信息
        home_team = match_data.get('homeTeam', match_data.get('homeTeamName', match_data.get('homeTeamId', '主队')))
        away_team = match_data.get('awayTeam', match_data.get('awayTeamName', match_data.get('awayTeamId', '客队')))
        context_parts.append(f"比赛球队: {home_team} vs {away_team}")
        
        # 3. 赔率信息 - 详细的菠菜公司赔率
        # 首先尝试从原始赔率数据字典中提取
        odds_data = match_data.get('odds', {})
        
        # 检查是否存在CSV格式的赔率字段
        has_csv_odds = any(key.endswith('_home_odds') for key in match_data.keys())
        
        if isinstance(odds_data, dict) and odds_data:
            # 从原始赔率数据字典中提取详细赔率
            initial_odds = self.get_initial_odds(odds_data)
            final_odds = self.get_final_odds(odds_data)
            
            # 分析赔率变化
            odds_change = self.analyze_odds_change(initial_odds, final_odds)
            
            # 遍历所有菠菜公司，显示详细赔率
            for company_id in self.bookmaker_map:
                if company_id in initial_odds and company_id in final_odds:
                    bookmaker_name = self.bookmaker_map[company_id]
                    initial = initial_odds[company_id]
                    final = final_odds[company_id]
                    change = odds_change.get(company_id, {})
                    
                    # 提取胜平负赔率
                    initial_home = float(initial[0])
                    initial_draw = float(initial[1])
                    initial_away = float(initial[2])
                    
                    final_home = float(final[0])
                    final_draw = float(final[1])
                    final_away = float(final[2])
                    
                    # 计算变化值
                    home_change = final_home - initial_home
                    draw_change = final_draw - initial_draw
                    away_change = final_away - initial_away
                    
                    # 添加到上下文
                    context_parts.append(f"{bookmaker_name}初赔(主/平/负): {initial_home:.2f}/{initial_draw:.2f}/{initial_away:.2f}")
                    context_parts.append(f"{bookmaker_name}终赔(主/平/负): {final_home:.2f}/{final_draw:.2f}/{final_away:.2f}")
                    context_parts.append(f"{bookmaker_name}赔率变化(主/平/负): {home_change:.2f}/{draw_change:.2f}/{away_change:.2f}")
        elif has_csv_odds:
            # 从CSV格式的赔率字段中提取
            # 支持的菠菜公司：ladbrokes等
            bookmaker_names = []
            for key in match_data.keys():
                if '_home_odds' in key:
                    bookmaker = key.replace('_home_odds', '')
                    bookmaker_names.append(bookmaker)
            
            for bookmaker in bookmaker_names:
                # 获取赔率数据
                home_odds = match_data.get(f'{bookmaker}_home_odds', 0)
                draw_odds = match_data.get(f'{bookmaker}_draw_odds', 0)
                away_odds = match_data.get(f'{bookmaker}_away_odds', 0)
                
                # 转换为中文名称
                if bookmaker.lower() == 'ladbrokes':
                    bookmaker_cn = '立博'
                else:
                    bookmaker_cn = bookmaker
                
                # 添加到上下文（CSV中只有终赔数据）
                context_parts.append(f"{bookmaker_cn}终赔(主/平/负): {float(home_odds):.2f}/{float(draw_odds):.2f}/{float(away_odds):.2f}")
        
        # 4. 球队本赛季数据
        # 从history中提取赛季数据
        history = match_data.get('history', {})
        # 确保history是字典类型
        if not isinstance(history, dict):
            history = {}
        home_season = history.get('homeSeasonData', {})
        away_season = history.get('awaySeasonData', {})
        # 确保home_season和away_season是字典类型
        if not isinstance(home_season, dict):
            home_season = {}
        if not isinstance(away_season, dict):
            away_season = {}
        
        # 主队本赛季数据
        home_rank = home_season.get('rank', '')
        home_win_rate = home_season.get('winRate', '')
        home_goals_for = home_season.get('goalsFor', '')
        home_goals_against = home_season.get('goalsAgainst', '')
        
        if home_rank or home_win_rate or home_goals_for or home_goals_against:
            win_rate_str = home_win_rate if '%' in str(home_win_rate) else f"{home_win_rate}%"
            context_parts.append(f"主队本赛季排名: {home_rank}, 胜率: {win_rate_str}, 进球: {home_goals_for}, 失球: {home_goals_against}")
        
        # 客队本赛季数据
        away_rank = away_season.get('rank', '')
        away_win_rate = away_season.get('winRate', '')
        away_goals_for = away_season.get('goalsFor', '')
        away_goals_against = away_season.get('goalsAgainst', '')
        
        if away_rank or away_win_rate or away_goals_for or away_goals_against:
            win_rate_str = away_win_rate if '%' in str(away_win_rate) else f"{away_win_rate}%"
            context_parts.append(f"客队本赛季排名: {away_rank}, 胜率: {win_rate_str}, 进球: {away_goals_for}, 失球: {away_goals_against}")
        
        # 5. 球队近期状态和近6场比赛
        # 首先尝试从直接特征中提取近期状态（来自MatchDataProcessor）
        home_wins = int(match_data.get('home_wins', match_data.get('home_recent_wins', 0)))
        home_draws = int(match_data.get('home_draws', match_data.get('home_recent_draws', 0)))
        home_losses = int(match_data.get('home_losses', match_data.get('home_recent_losses', 0)))
        home_form_string = match_data.get('home_form_string', '')
        
        away_wins = int(match_data.get('away_wins', match_data.get('away_recent_wins', 0)))
        away_draws = int(match_data.get('away_draws', match_data.get('away_recent_draws', 0)))
        away_losses = int(match_data.get('away_losses', match_data.get('away_recent_losses', 0)))
        away_form_string = match_data.get('away_form_string', '')
        
        # 如果直接特征中没有，尝试从history字段提取
        if not (home_wins or home_draws or home_losses):
            home_recent = history.get('homeData', [])
            home_last6 = self.calculate_last_n_matches(home_recent, num_matches=6)
            home_wins = home_last6['wins']
            home_draws = home_last6['draws']
            home_losses = home_last6['losses']
        
        if not (away_wins or away_draws or away_losses):
            away_recent = history.get('awayData', [])
            away_last6 = self.calculate_last_n_matches(away_recent, num_matches=6)
            away_wins = away_last6['wins']
            away_draws = away_last6['draws']
            away_losses = away_last6['losses']
        
        # 计算近期状态字符串
        if not home_form_string:
            home_form_str = self._generate_team_status(home_wins, home_draws, home_losses)
        else:
            home_form_str = home_form_string
        
        if not away_form_string:
            away_form_str = self._generate_team_status(away_wins, away_draws, away_losses)
        else:
            away_form_str = away_form_string
        
        context_parts.append(f"主队近期状态: {home_form_str}")
        context_parts.append(f"客队近期状态: {away_form_str}")
        context_parts.append(f"主队近6场: 胜{home_wins}场, 平{home_draws}场, 负{home_losses}场")
        context_parts.append(f"客队近6场: 胜{away_wins}场, 平{away_draws}场, 负{away_losses}场")
        
        # 6. 交锋历史
        # 首先尝试从直接特征中提取交锋历史（来自MatchDataProcessor）
        h2h_total = int(match_data.get('h2h_total_matches', 0))
        h2h_home_wins = int(match_data.get('h2h_home_wins', 0))
        h2h_away_wins = int(match_data.get('h2h_away_wins', 0))
        h2h_draws = int(match_data.get('h2h_draws', 0))
        
        # 如果直接特征中有交锋历史，直接使用
        if h2h_total > 0:
            h2h_str = f"共{h2h_total}场，主队{h2h_home_wins}胜，{h2h_draws}平，客队{h2h_away_wins}胜"
            context_parts.append(f"对战历史: {h2h_str}")
        else:
            # 否则从交锋历史详情中提取数据
            head_to_head = history.get('historyData', [])
            if head_to_head:
                h2h_stats = self.analyze_head_to_head(head_to_head)
                h2h_str = f"共{h2h_stats['total_matches']}场，主队{h2h_stats['home_wins']}胜，{h2h_stats['draws']}平，客队{h2h_stats['away_wins']}胜"
                context_parts.append(f"对战历史: {h2h_str}")
            else:
                # 尝试从head_to_head_history文本字段中提取
                h2h_text = match_data.get('head_to_head_history', '')
                if h2h_text:
                    context_parts.append(f"对战历史: {h2h_text}")
        
        # 7. 比赛结果（如果有且需要包含）
        if include_result:
            # 从赛季文件中获取赛果
            result = match_data.get('result')
            if result is not None:
                # 注意：result值可能有不同的含义，需要根据实际数据调整
                # 从prepare_data.py中看到：3: 主胜, 1: 平, 0: 客胜
                if result == 3:
                    result_str = '主队赢'
                elif result == 1:
                    result_str = '平局'
                elif result == 0:
                    result_str = '客队赢'
                else:
                    # 尝试从比分计算
                    home_score = match_data.get('homeScore', 0)
                    away_score = match_data.get('awayScore', 0)
                    if home_score > away_score:
                        result_str = '主队赢'
                    elif home_score < away_score:
                        result_str = '客队赢'
                    else:
                        result_str = '平局'
                context_parts.append(f"比赛结果: {result_str}")
            else:
                # 尝试从比分计算
                home_score = match_data.get('homeScore', 0)
                away_score = match_data.get('awayScore', 0)
                if home_score > away_score:
                    result_str = '主队赢'
                elif home_score < away_score:
                    result_str = '客队赢'
                elif home_score != 0 or away_score != 0:
                    result_str = '平局'
                # 否则不添加结果
        
        return '; '.join(context_parts)
    
    def generate_context_for_prediction(self, match_data: Dict[str, Any]) -> str:
        """生成用于预测的上下文"""
        # 预测上下文可以包含更详细的数值数据
        context_parts = []
        
        # 1. 基本信息
        match_id = match_data.get('matchId', '未知')
        context_parts.append(f"matchId={match_id}")
        
        # 2. 赔率信息 - 详细的菠菜公司赔率
        odds_data = match_data.get('odds', {})
        # 确保odds_data是字典类型
        if isinstance(odds_data, dict):
            initial_odds = self.get_initial_odds(odds_data)
            final_odds = self.get_final_odds(odds_data)
            
            # 遍历所有菠菜公司，显示详细赔率
            for company_id in self.bookmaker_map:
                if company_id in initial_odds and company_id in final_odds:
                    bookmaker_name = self.bookmaker_map[company_id].lower()
                    initial = initial_odds[company_id]
                    final = final_odds[company_id]
                    
                    context_parts.append(f"{bookmaker_name}_initial_home={initial[0]},{bookmaker_name}_initial_draw={initial[1]},{bookmaker_name}_initial_away={initial[2]}")
                    context_parts.append(f"{bookmaker_name}_final_home={final[0]},{bookmaker_name}_final_draw={final[1]},{bookmaker_name}_final_away={final[2]}")
        
        # 3. 球队赛季数据 - 本赛季
        history = match_data.get('history', {})
        # 确保history是字典类型
        if isinstance(history, dict):
            home_season = history.get('homeSeasonData', {})
            away_season = history.get('awaySeasonData', {})
        
        if home_season:
            context_parts.append(f"home_rank={home_season.get('rank', '0')},home_win_rate={home_season.get('winRate', '0').replace('%', '')}")
            context_parts.append(f"home_goals_for={home_season.get('goalsFor', '0')},home_goals_against={home_season.get('goalsAgainst', '0')}")
        
        if away_season:
            context_parts.append(f"away_rank={away_season.get('rank', '0')},away_win_rate={away_season.get('winRate', '0').replace('%', '')}")
            context_parts.append(f"away_goals_for={away_season.get('goalsFor', '0')},away_goals_against={away_season.get('goalsAgainst', '0')}")
        
        # 4. 近6场比赛统计
        home_recent = history.get('homeData', [])
        away_recent = history.get('awayData', [])
        
        if home_recent:
            home_last6 = self.calculate_last_n_matches(home_recent, num_matches=6)
            context_parts.append(f"home_last6_wins={home_last6['wins']},home_last6_draws={home_last6['draws']},home_last6_losses={home_last6['losses']}")
        
        if away_recent:
            away_last6 = self.calculate_last_n_matches(away_recent, num_matches=6)
            context_parts.append(f"away_last6_wins={away_last6['wins']},away_last6_draws={away_last6['draws']},away_last6_losses={away_last6['losses']}")
        
        # 5. 近期状态（胜率和进球）
        if home_recent:
            home_form = self.calculate_recent_form(home_recent)
            context_parts.append(f"home_recent_win_rate={home_form['win_rate']:.2f},home_goals_per_match={home_form['goals_per_match']:.2f}")
        
        if away_recent:
            away_form = self.calculate_recent_form(away_recent)
            context_parts.append(f"away_recent_win_rate={away_form['win_rate']:.2f},away_goals_per_match={away_form['goals_per_match']:.2f}")
        
        # 6. 交锋历史
        head_to_head = history.get('historyData', [])
        if head_to_head:
            h2h_stats = self.analyze_head_to_head(head_to_head)
            context_parts.append(f"h2h_total={h2h_stats['total_matches']},h2h_home_wins={h2h_stats['home_wins']},h2h_away_wins={h2h_stats['away_wins']},h2h_draws={h2h_stats['draws']}")
        
        return ','.join(context_parts)
    
    def generate_context(self, match_data: Dict[str, Any], context_type: str = 'knowledge_matching', include_result: bool = False) -> str:
        """生成上下文
        
        Args:
            match_data: 比赛数据
            context_type: 上下文类型，可选值：knowledge_matching, prediction
            include_result: 是否包含赛果信息
            
        Returns:
            上下文字符串
        """
        try:
            # 确保match_data是字典类型
            if not isinstance(match_data, dict):
                print(f"生成上下文失败: match_data不是字典类型，而是{type(match_data)}")
                return ""
            
            # 确保所有可能的嵌套字段都是字典类型
            if 'odds' in match_data and not isinstance(match_data['odds'], dict):
                match_data['odds'] = {}
            if 'history' in match_data and not isinstance(match_data['history'], dict):
                match_data['history'] = {}
            if 'homeSeasonData' in match_data and not isinstance(match_data['homeSeasonData'], dict):
                match_data['homeSeasonData'] = {}
            if 'awaySeasonData' in match_data and not isinstance(match_data['awaySeasonData'], dict):
                match_data['awaySeasonData'] = {}
            
            if context_type == 'knowledge_matching':
                return self.generate_context_for_knowledge_matching(match_data, include_result)
            elif context_type == 'prediction':
                return self.generate_context_for_prediction(match_data)
            else:
                raise ValueError(f"不支持的上下文类型: {context_type}")
        except Exception as e:
            print(f"生成上下文失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_context_statistics(self, context: str) -> Dict[str, Any]:
        """获取上下文统计信息"""
        return {
            'length': len(context),
            'words': len(context.split()),
            'dimensions': context.count(';') + 1 if ';' in context else 1,
            'has_odds_info': '赔率' in context or '初赔' in context or '终赔' in context,
            'has_team_info': '排名' in context or '胜率' in context or '进球' in context,
            'has_history_info': '交锋' in context or '近期' in context
        }

# 测试示例
if __name__ == "__main__":
    generator = ContextGenerator()
    
    # 加载示例比赛数据
    sample_file = "/Users/Williamhiler/Documents/my-project/train/original-data/2017-2018/details/R_1/1394661.json"
    match_data = generator.load_match_data(sample_file)
    
    # 生成知识匹配上下文
    knowledge_context = generator.generate_context(match_data, 'knowledge_matching')
    print("=== 知识匹配上下文 ===")
    print(knowledge_context)
    print("\n上下文统计:")
    print(generator.get_context_statistics(knowledge_context))
    
    # 生成预测上下文
    prediction_context = generator.generate_context(match_data, 'prediction')
    print("\n=== 预测上下文 ===")
    print(prediction_context)
    print("\n上下文统计:")
    print(generator.get_context_statistics(prediction_context))
