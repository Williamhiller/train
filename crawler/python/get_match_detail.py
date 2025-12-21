#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
抓取比赛详情，包括历史交锋和赔率信息
输入参数：matchId
自动生成历史交锋和赔率信息的URL
"""

import asyncio
import re
import os
import json
from get_all_match import get_all_match

# 公司名称映射
company = {
    82: 'Ladbrokes',
    115: 'william'
}

# 格式化函数：将数值保留两位小数
def format_decimal(value):
    return f"{float(value):.2f}"

# 数据解析函数：从原始数据数组中提取所需字段
def parse_match_data(match_array):
    return [
        match_array[0],     # 比赛时间
        match_array[1],     # 联赛id
        match_array[4],     # 主队id
        match_array[6],     # 客队id
        match_array[8],     # 主队进球数
        match_array[9],     # 客队进球数
        match_array[12]     # 赛果
    ]

# 解析赔率数据并进行结构调整
# season参数：赛季信息，如"2023-2024"
def parse_odds_data(detail, season):
    detail_parts = detail.split('^')
    if len(detail_parts) < 2:
        print('赔率详情格式错误')
        return []

    odds_data = detail_parts[1]
    odds_array = [item for item in odds_data.split(';') if item.strip()]

    # 解析赛季信息，获取开始年份和结束年份
    season_years = season.split('-')
    start_year = int(season_years[0])
    end_year = int(season_years[1])

    result = []
    for odds in odds_array:
        parts = odds.split('|')
        
        # 解析时间部分，获取月份
        time_part = parts[3]
        month_match = re.match(r'(\d{2})-(\d{2})', time_part)
        
        year = end_year
        if month_match:
            month = int(month_match.group(1))
            # 根据月份确定年份：大于6月使用开始年份，否则使用结束年份
            year = start_year if month > 5 else end_year
        
        full_time = f"{year}-{time_part}"

        # 当parts长度不足7时，胜平负赔付比例默认使用0
        result.append([
            format_decimal(parts[0]),  # 胜赔率（保留两位小数）
            format_decimal(parts[1]),  # 平赔率（保留两位小数）
            format_decimal(parts[2]),  # 负赔率（保留两位小数）
            format_decimal(parts[4] if len(parts) > 4 else 0),  # 胜赔付比例（保留两位小数）
            format_decimal(parts[5] if len(parts) > 5 else 0),  # 平赔付比例（保留两位小数）
            format_decimal(parts[6] if len(parts) > 6 else 0),  # 负赔付比例（保留两位小数）
            full_time   # 完整时间（包含年份）
        ])
    
    return result

# 处理数组中的每个子项，确保子项内部没有换行
def process_data_array(data_array):
    data_strings = [json.dumps(item, ensure_ascii=False).replace('\n', '').replace('\s+', ' ') for item in data_array]
    return '\n    ' + ',\n    '.join(data_strings) + '\n  '

# 主函数：获取比赛详情
# save_path 可选参数，指定保存路径，如果不提供则不保存到文件
# season 可选参数，指定赛季信息，如果不提供则使用默认值2023-2024
async def get_match_detail(match_id, save_path=None, season=None):
    print(f"开始获取比赛详情，matchId: {match_id}")
    
    # 根据matchId生成对应的URL
    history_url = f"https://zq.titan007.com/analysis/{match_id}cn.htm"
    odd_url = f"https://1x2.titan007.com/oddslist/{match_id}.htm"
    
    print(f"历史交锋URL: {history_url}")
    print(f"赔率信息URL: {odd_url}")
    
    # 同时获取历史交锋数据和赔率数据
    print('开始获取历史交锋数据...')
    history_data = await get_all_match(history_url, 'analyze')
    
    print('开始获取赔率数据...')
    odd_data = await get_all_match(odd_url, 'odd')
    
    # 检查数据是否获取成功
    if not history_data or not odd_data:
        print('未能获取到完整的比赛数据')
        return None
    
    # 处理历史交锋数据
    away_data = history_data.get('awayData', [])
    home_data = history_data.get('homeData', [])
    raw_history_data = history_data.get('historyData', [])
    
    # 解析历史交锋数据
    parsed_away_data = [parse_match_data(item) for item in away_data]
    parsed_home_data = [parse_match_data(item) for item in home_data]
    parsed_history_data = [parse_match_data(item) for item in raw_history_data]
    
    # 处理赔率数据
    game = odd_data.get('game', [])
    game_detail = odd_data.get('gameDetail', [])
    
    # 遍历game数组，找到82和115的项
    target_odd_ids = {}
    for item in game:
        parts = item.split('|')
        if len(parts) >= 2 and parts[0] in ['82', '115']:
            target_odd_ids[parts[0]] = parts[1]
    
    # 检查是否找到82和115的项
    if '82' not in target_odd_ids or '115' not in target_odd_ids:
        print('未找到目标赔率ID')
        return None
    
    # 遍历gameDetail数组，找到对应的项
    target_details = {}
    for item in game_detail:
        for key, value in target_odd_ids.items():
            if item.startswith(f"{value}^"):
                target_details[key] = item
                break
    
    # 检查是否找到对应的项
    if '82' not in target_details or '115' not in target_details:
        print('未找到目标赔率详情')
        return None
    
    # 解析82和115的赔率数据，使用传入的赛季信息或默认值
    current_season = season or '2023-2024'
    parsed_odds_82 = parse_odds_data(target_details['82'], current_season)
    parsed_odds_115 = parse_odds_data(target_details['115'], current_season)
    
    # 构建完整的结果对象
    result = {
        'matchId': match_id,
        'history': {
            'awayData': parsed_away_data,
            'homeData': parsed_home_data,
            'historyData': parsed_history_data
        },
        'odds': {
            'oddId': {
                '82': target_odd_ids['82'],
                '115': target_odd_ids['115']
            },
            '82': parsed_odds_82,
            '115': parsed_odds_115
        }
    }
    
    # 构建完整的JSON字符串，保持与JS版本相同的风格
    away_data_content = process_data_array(result['history']['awayData'])
    home_data_content = process_data_array(result['history']['homeData'])
    history_data_content = process_data_array(result['history']['historyData'])
    odds_82_content = process_data_array(result['odds']['82'])
    odds_115_content = process_data_array(result['odds']['115'])

    json_str = f'''
{{
  "matchId": "{result['matchId']}",
  "history": {{
    "awayData": [{away_data_content}],
    "homeData": [{home_data_content}],
    "historyData": [{history_data_content}]
  }},
  "odds": {{
    "oddId": {{
      "82": "{result['odds']['oddId']['82']}",
      "115": "{result['odds']['oddId']['115']}"
    }},
    "82": [{odds_82_content}],
    "115": [{odds_115_content}]
  }}
}}
'''
    
    # 如果提供了保存路径，则保存到文件
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f'比赛详情数据已保存到: {save_path}')
    
    return result

if __name__ == "__main__":
    # 从命令行参数获取matchId，如果没有提供则使用默认值
    import sys
    match_id = sys.argv[1] if len(sys.argv) > 1 else '2590911'
    
    print(f"正在测试matchId: {match_id}")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_match_detail(match_id))
