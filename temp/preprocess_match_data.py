#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 处理examples目录下的比赛数据

功能：
1. 加载examples目录下的所有比赛数据文件
2. 提取特征：
   - 基本信息（比赛时间、对阵、比分）
   - 赔率信息
   - 球队历史数据
   - 赛季数据
3. 构建训练数据集
4. 保存为训练格式（指令-回答对）
5. 支持多种输出格式

使用方法：
python preprocess_match_data.py
"""

import json
import os
import argparse
from datetime import datetime

# ==================== 配置参数 ====================
class Config:
    # 输入数据路径
    INPUT_DIR = "/Users/Williamhiler/Documents/my-project/train/examples"
    
    # 输出数据路径
    OUTPUT_DIR = "/Users/Williamhiler/Documents/my-project/train/colab_training/match"
    
    # 输出文件名
    OUTPUT_FILE = "match_train_data.json"
    
    # 输出格式选项：
    # - "instruction"：指令-回答格式
    # - "chat"：对话格式
    # - "text"：纯文本格式
    OUTPUT_FORMAT = "instruction"
    
    # 是否保存样本统计信息
    SAVE_STATISTICS = True
    
    # 统计信息文件名
    STATISTICS_FILE = "data_statistics.json"

# ==================== 映射字典 ====================

# 庄家ID到名称的映射
bookie_mapping = {
    "82": "立博",
    "115": "威廉",
    "281": "bet365",
    "2": "必发"
}

# 结果映射（310制）
result_mapping = {
    "3": "胜",  # 历史数据中可能使用3表示胜
    "1": "平",  # 历史数据中可能使用1表示平
    "0": "负"   # 历史数据中可能使用0表示负
}

# 球队ID到名称的映射（基础映射，会动态扩展）
team_mapping = {
    # 英超球队
    "19": "曼联",
    "62": "曼城",
    "20": "阿森纳",
    "27": "切尔西",
    "34": "利物浦",
    "30": "热刺",
    "26": "埃弗顿",
    "33": "莱斯特城",
    "59": "莱斯特城",  # 可能存在重复ID
    "35": "南安普顿",
    "53": "纽卡斯尔",
    "24": "西汉姆联",
    "25": "水晶宫",
    "65": "斯托克城",
    "58": "沃特福德",
    "18": "伯恩茅斯",
    "1194": "斯旺西",
    "28": "桑德兰",
    "348": "诺维奇",
    "82": "阿斯顿维拉",
    "87": "西布朗",
    "32": "狼队",
    "68": "布伦特福德",
    "57": "富勒姆",
    "31": "伯恩利",
    "23": "埃弗顿",  # 可能的备用ID
    "21": "南安普顿",  # 可能的备用ID
    
    # 英冠球队
    "17": "米德尔斯堡",
    "36": "布莱顿",
    "47": "哈德斯菲尔德",
    "50": "德比郡",
    "55": "伯明翰",
    "69": "利兹联",
    "100": "诺丁汉森林",
    "107": "雷丁",
    "117": "谢菲尔德联",
    "126": "女王公园巡游者",
    "186": "普雷斯顿",
    "215": "卡迪夫城",
    "221": "布莱克本",
    "358": "斯旺西",
    "384": "赫尔城",
    "486": "布里斯托尔城",
    "616": "伊普斯维奇",
    "1186": "诺维奇",
    "1199": "西布罗姆维奇",
    "1201": "布莱克本",
    "1208": "富勒姆",
    "1220": "伯明翰",
    "1840": "谢周三",
    "1863": "斯托克城",
    "1866": "女王公园",
    "3280": "伯恩利",
    "3427": "巴恩斯利",
    "344": "查尔顿",
    "3545": "卢顿",
    "8521": "维冈竞技",
    "9194": "维冈",
    "11011": "博尔顿",
    "51": "布莱顿",  # 可能的备用ID
    "379": "布里斯托尔城",  # 可能的备用ID
    "336": "谢菲尔德联",  # 可能的备用ID
    "1207": "雷丁",  # 可能的备用ID
    "1187": "德比郡",  # 可能的备用ID
    "1188": "斯旺西",  # 可能的备用ID
    "1190": "米德尔斯堡",  # 可能的备用ID
    "1199": "西布罗姆维奇",  # 可能的备用ID
    "1201": "布莱克本",  # 可能的备用ID
    "1202": "卡迪夫城",  # 可能的备用ID
    "1203": "普雷斯顿",  # 可能的备用ID
    "1264": "谢周三",  # 可能的备用ID
    "1863": "斯托克城",  # 可能的备用ID
    "1866": "女王公园巡游者",  # 可能的备用ID
    "3280": "伯恩利",  # 可能的备用ID
    "3427": "巴恩斯利",  # 可能的备用ID
    "344": "查尔顿",  # 可能的备用ID
    "3545": "卢顿",  # 可能的备用ID
    "8521": "维冈竞技",  # 可能的备用ID
    "9194": "维冈",  # 可能的备用ID
    "11011": "博尔顿"  # 可能的备用ID
}

# 动态球队名称缓存，用于存储从数据中提取的球队名称
dynamic_team_cache = {}

# ==================== 辅助函数 ====================

def get_team_name(team_id):
    """获取球队名称
    
    Args:
        team_id: 球队ID
        
    Returns:
        str: 球队名称
    """
    team_id_str = str(team_id)
    
    # 先从基础映射查找
    if team_id_str in team_mapping:
        return team_mapping[team_id_str]
    
    # 再从动态缓存查找
    if team_id_str in dynamic_team_cache:
        return dynamic_team_cache[team_id_str]
    
    # 如果都找不到，返回原始ID，但添加到动态缓存以便后续处理
    dynamic_team_cache[team_id_str] = team_id_str
    return team_id_str


def get_bookie_name(bookie_id):
    """获取庄家名称
    
    Args:
        bookie_id: 庄家ID
        
    Returns:
        str: 庄家名称
    """
    return bookie_mapping.get(bookie_id, bookie_id)


def normalize_team_history(team_data):
    """归一化球队历史数据
    
    将球队历史数据转换为胜平负格式，数组最后一位为结果（310制）
    
    Args:
        team_data: 球队历史数据
        
    Returns:
        list: 归一化后的球队历史数据
    """
    normalized_data = []
    if team_data and isinstance(team_data, list):
        for match in team_data:
            if isinstance(match, list) and len(match) >= 4:
                # 转换为 [时间, 对手名称, 主队/客队, 结果] 格式
                opponent = get_team_name(match[1])
                timestamp = match[0]
                is_home = "主" if match[2] == 1 else "客"
                # 结果映射：3=胜，1=平，0=负
                result = result_mapping.get(str(match[3]), str(match[3]))
                normalized_data.append(f"{timestamp} {opponent} {is_home} {result}")
    return normalized_data


def analyze_odds_changes(odds_list):
    """分析赔率变化，提供更丰富的赔率变化信息
    
    Args:
        odds_list: 赔率列表
        
    Returns:
        dict: 详细的赔率变化分析
    """
    if not odds_list or len(odds_list) < 2:
        return {
            "change": "无变化", 
            "direction": "", 
            "magnitude": 0,
            "initial_odds": [],
            "final_odds": [],
            "trend": "稳定",
            "significant_changes": 0,
            "home_trend": "稳定",
            "draw_trend": "稳定",
            "away_trend": "稳定"
        }
    
    # 获取初始和最终赔率
    initial = odds_list[0]
    final = odds_list[-1]
    
    if len(initial) < 3 or len(final) < 3:
        return {
            "change": "无变化", 
            "direction": "", 
            "magnitude": 0,
            "initial_odds": [],
            "final_odds": [],
            "trend": "稳定",
            "significant_changes": 0,
            "home_trend": "稳定",
            "draw_trend": "稳定",
            "away_trend": "稳定"
        }
    
    # 转换为浮点数
    initial_home = float(initial[0])
    initial_draw = float(initial[1])
    initial_away = float(initial[2])
    final_home = float(final[0])
    final_draw = float(final[1])
    final_away = float(final[2])
    
    # 计算变化
    home_change = final_home - initial_home
    draw_change = final_draw - initial_draw
    away_change = final_away - initial_away
    
    # 确定变化最大的方向
    max_change = max(abs(home_change), abs(draw_change), abs(away_change))
    
    # 计算趋势
    trends = {}
    for odd_type, change in [
        ("home", home_change),
        ("draw", draw_change),
        ("away", away_change)
    ]:
        if abs(change) < 0.05:
            trends[f"{odd_type}_trend"] = "稳定"
        elif change > 0:
            trends[f"{odd_type}_trend"] = "上升"
        else:
            trends[f"{odd_type}_trend"] = "下降"
    
    # 计算显著变化次数
    significant_changes = 0
    for odd_list in odds_list:
        if len(odd_list) >= 3:
            # 检查是否有显著变化（大于0.1）
            if any(abs(float(odd) - float(initial[i])) > 0.1 for i, odd in enumerate(odd_list[:3])):
                significant_changes += 1
    
    # 整体趋势
    if all(trend == "稳定" for trend in trends.values()):
        overall_trend = "稳定"
    elif trends["home_trend"] == "下降" and trends["away_trend"] == "上升":
        overall_trend = "倾向主胜"
    elif trends["home_trend"] == "上升" and trends["away_trend"] == "下降":
        overall_trend = "倾向客胜"
    elif trends["draw_trend"] == "下降":
        overall_trend = "倾向平局"
    else:
        overall_trend = "波动"
    
    # 方向描述
    if max_change < 0.05:
        direction = ""
        change_type = "无明显变化"
    else:
        change_type = "有变化"
        if max(home_change, draw_change, away_change) == home_change:
            direction = "主胜赔率上升"
        elif max(home_change, draw_change, away_change) == draw_change:
            direction = "平局赔率上升"
        else:
            direction = "客胜赔率上升"
        
        if min(home_change, draw_change, away_change) == home_change:
            direction += "/主胜赔率下降"
        elif min(home_change, draw_change, away_change) == draw_change:
            direction += "/平局赔率下降"
        else:
            direction += "/客胜赔率下降"
    
    return {
        "change": change_type,
        "direction": direction,
        "magnitude": max_change,
        "initial_odds": [initial_home, initial_draw, initial_away],
        "final_odds": [final_home, final_draw, final_away],
        "trend": overall_trend,
        "significant_changes": significant_changes,
        **trends
    }


def load_match_data(file_path_or_dir):
    """加载比赛数据
    
    支持两种模式：
    1. 加载单个文件
    2. 加载目录下所有_aggregated.json结尾的文件并合并
    
    Args:
        file_path_or_dir: 文件路径或目录路径
        
    Returns:
        dict: 合并后的比赛数据
    """
    print(f"\n1. 加载比赛数据：{file_path_or_dir}")
    
    # 检查是文件还是目录
    if os.path.isfile(file_path_or_dir):
        # 加载单个文件
        with open(file_path_or_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✓ 成功加载单个文件")
        print(f"     文件：{os.path.basename(file_path_or_dir)}")
        print(f"     比赛场次：{len(data)}")
        return data
    elif os.path.isdir(file_path_or_dir):
        # 加载目录下所有符合条件的文件
        merged_data = {}
        file_count = 0
        
        # 遍历目录下所有文件
        for filename in sorted(os.listdir(file_path_or_dir)):
            if filename.endswith('_aggregated.json'):
                file_path = os.path.join(file_path_or_dir, filename)
                print(f"   - 加载文件：{filename}")
                
                # 加载单个文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 合并数据
                merged_data.update(data)
                file_count += 1
        
        print(f"   ✓ 成功加载 {file_count} 个文件")
        print(f"     总比赛场次：{len(merged_data)}")
        return merged_data
    else:
        raise ValueError(f"{file_path_or_dir} 不是有效的文件或目录路径")


def extract_features(match_id, match_info):
    """从比赛信息中提取特征
    
    Args:
        match_id: 比赛ID
        match_info: 比赛详细信息
        
    Returns:
        dict: 提取的特征
    """
    # 基本信息
    match_time = match_info.get("matchTime", "")
    home_team_id = match_info.get("homeTeamId", "")
    away_team_id = match_info.get("awayTeamId", "")
    
    # 转换球队ID为名称
    home_team = get_team_name(home_team_id)
    away_team = get_team_name(away_team_id)
    
    result = match_info.get("result", "")
    home_score = match_info.get("homeScore", 0)
    away_score = match_info.get("awayScore", 0)
    
    # 提取赔率信息
    odds_info = []
    details = match_info.get("details", {})
    odds = details.get("odds", {})
    for bookie_id, odds_list in odds.items():
        if odds_list and isinstance(odds_list, list):
            latest_odds = odds_list[-1]  # 获取最新赔率
            if len(latest_odds) >= 3:
                # 分析赔率变化
                odds_change = analyze_odds_changes(odds_list)
                odds_info.append({
                    "bookie_id": bookie_id,
                    "bookie_name": get_bookie_name(bookie_id),
                    "home_win": latest_odds[0],
                    "draw": latest_odds[1],
                    "away_win": latest_odds[2],
                    "timestamp": latest_odds[-1] if len(latest_odds) > 3 else "",
                    "change_analysis": odds_change
                })
    
    # 提取球队历史数据并归一化
    history = details.get("history", {})
    home_data = normalize_team_history(history.get("homeData", []))
    away_data = normalize_team_history(history.get("awayData", []))
    history_data = history.get("historyData", [])
    
    # 提取赛季数据
    home_season = history.get("homeSeasonData", {})
    away_season = history.get("awaySeasonData", {})
    
    # 分析历史交锋数据
    home_win_h2h = 0
    draw_h2h = 0
    away_win_h2h = 0
    total_h2h = 0
    last_5_h2h = []
    
    if history_data and isinstance(history_data, list):
        total_h2h = len(history_data)
        
        for match in history_data:
            if isinstance(match, list) and len(match) >= 6:
                # 历史交锋数据格式：[主队, 客队, 主队得分, 客队得分, 结果, 时间]
                h2h_home_team = match[0]
                h2h_away_team = match[1]
                h2h_home_score = int(match[2]) if isinstance(match[2], (int, str)) else 0
                h2h_away_score = int(match[3]) if isinstance(match[3], (int, str)) else 0
                h2h_result = match[4]
                h2h_time = match[5]
                
                # 统计胜负平
                if h2h_result == 3:  # 主队胜
                    if h2h_home_team == home_team_id:
                        home_win_h2h += 1
                        last_5_h2h.append("主胜")
                    else:
                        away_win_h2h += 1
                        last_5_h2h.append("客胜")
                elif h2h_result == 1:  # 平局
                    draw_h2h += 1
                    last_5_h2h.append("平局")
                elif h2h_result == 0:  # 客队胜
                    if h2h_away_team == home_team_id:
                        home_win_h2h += 1
                        last_5_h2h.append("主胜")
                    else:
                        away_win_h2h += 1
                        last_5_h2h.append("客胜")
        
        # 只保留最近5场交锋
        last_5_h2h = last_5_h2h[-5:]
    
    # 构建特征文本
    features_text = f"比赛时间：{match_time}\n"
    features_text += f"对阵：{home_team} VS {away_team}\n"
    features_text += f"比赛结果：{home_score}-{away_score}（{['客胜', '平局', '未知', '主胜'][result] if result in [0,1,3] else '未知'}\n"    
    # 添加历史交锋数据
    if total_h2h > 0:
        features_text += f"历史交锋：共{total_h2h}次交锋，{home_team}胜{home_win_h2h}场，平局{draw_h2h}场，{away_team}胜{away_win_h2h}场\n"
        if last_5_h2h:
            features_text += f"最近5场交锋：{', '.join(last_5_h2h)}\n"
    
    # 赔率信息
    if odds_info:
        odds_str = []
        for odd in odds_info[:3]:  # 只取前3个庄家的赔率
            change_analysis = odd['change_analysis']
            odds_str.append(f"{odd['bookie_name']}：胜{odd['home_win']}，平{odd['draw']}，负{odd['away_win']}（趋势：{change_analysis['trend']}，主胜赔率{change_analysis['home_trend']}，平局赔率{change_analysis['draw_trend']}，客胜赔率{change_analysis['away_trend']}，显著变化{change_analysis['significant_changes']}次）")
        features_text += f"赔率信息：{'; '.join(odds_str)}\n"
    
    # 球队近期战绩
    if home_data:
        features_text += f"主队近期战绩：{str(home_data[:6])}\n"
    if away_data:
        features_text += f"客队近期战绩：{str(away_data[:6])}\n"
    
    # 赛季数据
    if home_season:
        features_text += f"主队赛季数据：{str(home_season)}\n"
    if away_season:
        features_text += f"客队赛季数据：{str(away_season)}\n"
    
    return {
        "match_time": match_time,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team": home_team,
        "away_team": away_team,
        "result": result,
        "home_score": home_score,
        "away_score": away_score,
        "odds_info": odds_info,
        "home_data": home_data,
        "away_data": away_data,
        "history_data": history_data,
        "home_season": home_season,
        "away_season": away_season,
        "features_text": features_text
    }


def build_training_dataset(match_data, output_format="instruction"):
    """构建训练数据集
    
    Args:
        match_data: 比赛数据字典
        output_format: 输出格式
        
    Returns:
        list: 训练数据集
        dict: 统计信息
    """
    print(f"\n2. 构建训练数据集")
    print(f"   输出格式：{output_format}")
    
    training_samples = []
    statistics = {
        "total_matches": len(match_data),
        "total_samples": 0,
        "home_win_count": 0,
        "draw_count": 0,
        "away_win_count": 0,
        "output_format": output_format,
        "created_at": datetime.now().isoformat()
    }
    
    # 遍历所有比赛数据
    for i, (match_id, match_info) in enumerate(match_data.items()):
        # 提取特征
        features = extract_features(match_id, match_info)
        
        # 统计比赛结果
        result = features["result"]
        
        # 使用result字段直接统计结果，对应关系：3=主胜，1=平局，0=客胜
        if result == 3:
            statistics["home_win_count"] += 1
        elif result == 1:
            statistics["draw_count"] += 1
        elif result == 0:
            statistics["away_win_count"] += 1
        
        # 构建指令和期望输出
        instruction = f"请基于以下比赛数据，分析这场比赛的赔率变化和球队状态，并预测比赛结果。\n\n{features['features_text']}"
        
        # 构建回答
        answer = f"根据比赛数据和赔率分析，这场比赛的结果是{features['home_team']} {features['home_score']}-{features['away_score']} {features['away_team']}，最终结果为{['客胜', '平局', '未知', '主胜'][result] if result in [0,1,3] else '未知'}。\n\n"
        
        # 历史交锋分析
        if features['history_data']:
            # 统计历史交锋数据
            home_win_h2h = 0
            draw_h2h = 0
            away_win_h2h = 0
            total_h2h = len(features['history_data'])
            
            for match in features['history_data']:
                if isinstance(match, list) and len(match) >= 5:
                    h2h_result = match[4]
                    if h2h_result == 3:  # 主队胜
                        if match[0] == features['home_team_id']:
                            home_win_h2h += 1
                        else:
                            away_win_h2h += 1
                    elif h2h_result == 1:  # 平局
                        draw_h2h += 1
                    elif h2h_result == 0:  # 客队胜
                        if match[1] == features['home_team_id']:
                            home_win_h2h += 1
                        else:
                            away_win_h2h += 1
            
            answer += f"历史交锋分析：\n"
            answer += f"  双方共交锋{total_h2h}次\n"
            answer += f"  {features['home_team']}胜{home_win_h2h}场，平局{draw_h2h}场，{features['away_team']}胜{away_win_h2h}场\n"
            if features['history_data'][:5]:
                answer += f"  最近5次交锋：\n"
                for i, match in enumerate(features['history_data'][:5]):
                    if isinstance(match, list) and len(match) >= 6:
                        answer += f"    {i+1}. {match[5]} {match[0]} {match[1]} {match[2]}-{match[3]} {['客胜', '平局', '未知', '主胜'][match[4]] if match[4] in [0,1,3] else '未知'}\n"
        
        # 赔率分析
        if features['odds_info']:
            odds_analysis = "赔率分析：\n"
            for odd in features['odds_info'][:3]:
                change_analysis = odd['change_analysis']
                odds_analysis += f"  {odd['bookie_name']}：胜{odd['home_win']}，平{odd['draw']}，负{odd['away_win']}\n"
                odds_analysis += f"    趋势：{change_analysis['trend']}\n"
                odds_analysis += f"    主胜赔率：{change_analysis['home_trend']}，平局赔率：{change_analysis['draw_trend']}，客胜赔率：{change_analysis['away_trend']}\n"
                odds_analysis += f"    显著变化次数：{change_analysis['significant_changes']}\n"
                if change_analysis['initial_odds']:
                    odds_analysis += f"    初始赔率：主胜{change_analysis['initial_odds'][0]:.2f}，平局{change_analysis['initial_odds'][1]:.2f}，客胜{change_analysis['initial_odds'][2]:.2f}\n"
                    odds_analysis += f"    最终赔率：主胜{change_analysis['final_odds'][0]:.2f}，平局{change_analysis['final_odds'][1]:.2f}，客胜{change_analysis['final_odds'][2]:.2f}\n"
            answer += odds_analysis
        
        # 球队状态分析
        answer += "球队状态分析：\n"
        if features['home_data']:
            answer += f"  主队近期6场战绩：{str(features['home_data'][:6])}\n"
        if features['away_data']:
            answer += f"  客队近期6场战绩：{str(features['away_data'][:6])}\n"
        
        # 赛季数据
        if features['home_season']:
            answer += f"  主队赛季数据：{str(features['home_season'])}\n"
        if features['away_season']:
            answer += f"  客队赛季数据：{str(features['away_season'])}\n"
        
        # 根据输出格式构建样本
        if output_format == "instruction":
            # 指令-回答格式
            sample = {
                "text": f"### 指令：\n{instruction}\n\n### 回答：\n{answer}"
            }
        elif output_format == "chat":
            # 对话格式
            sample = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer}
                ]
            }
        elif output_format == "text":
            # 纯文本格式
            sample = {
                "instruction": instruction,
                "answer": answer
            }
        else:
            raise ValueError(f"不支持的输出格式：{output_format}")
        
        training_samples.append(sample)
        
        # 进度显示
        if (i + 1) % 500 == 0:
            print(f"   处理进度：{i + 1}/{len(match_data)} 场比赛")
    
    statistics["total_samples"] = len(training_samples)
    
    print(f"   ✓ 成功构建训练数据集")
    print(f"     样本数量：{len(training_samples)}")
    print(f"     主胜：{statistics['home_win_count']}场")
    print(f"     平局：{statistics['draw_count']}场")
    print(f"     客胜：{statistics['away_win_count']}场")
    
    return training_samples, statistics


def save_processed_data(data, output_file):
    """保存处理后的数据
    
    Args:
        data: 处理后的数据
        output_file: 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n3. 数据保存完成")
    print(f"   输出文件：{output_file}")
    print(f"   文件大小：{os.path.getsize(output_file) / 1024:.2f} KB")


def save_statistics(statistics, output_file):
    """保存统计信息
    
    Args:
        statistics: 统计信息
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"   统计信息：{output_file}")

# ==================== 主函数 ====================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据预处理脚本 - 处理examples目录下的比赛数据')
    parser.add_argument('--input-dir', type=str, default=Config.INPUT_DIR, help='输入数据目录')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR, help='输出数据目录')
    parser.add_argument('--output-format', type=str, default=Config.OUTPUT_FORMAT, 
                        choices=['instruction', 'chat', 'text'], help='输出格式')
    parser.add_argument('--save-statistics', action='store_true', default=Config.SAVE_STATISTICS, help='保存统计信息')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据预处理脚本")
    print("=" * 60)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        match_data = load_match_data(args.input_dir)
        
        # 2. 构建训练数据集
        training_samples, statistics = build_training_dataset(match_data, args.output_format)
        
        # 3. 保存数据
        output_file = os.path.join(args.output_dir, Config.OUTPUT_FILE)
        save_processed_data(training_samples, output_file)
        
        # 4. 保存统计信息
        if args.save_statistics:
            statistics_file = os.path.join(args.output_dir, Config.STATISTICS_FILE)
            save_statistics(statistics, statistics_file)
        
        print("=" * 60)
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 数据预处理完成！")
        print("=" * 60)
        print(f"📊 统计信息：")
        print(f"   总比赛场次：{statistics['total_matches']}")
        print(f"   生成样本数：{statistics['total_samples']}")
        print(f"   主胜比例：{statistics['home_win_count'] / statistics['total_matches']:.2%}")
        print(f"   平局比例：{statistics['draw_count'] / statistics['total_matches']:.2%}")
        print(f"   客胜比例：{statistics['away_win_count'] / statistics['total_matches']:.2%}")
        print(f"   输出格式：{statistics['output_format']}")
        print(f"   输出文件：{output_file}")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print("❌ 数据预处理失败！")
        print(f"错误信息：{str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

# ==================== 入口 ====================
if __name__ == "__main__":
    main()