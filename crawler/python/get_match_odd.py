#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
获取比赛赔率数据
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

async def get_match_odd():
    # 从URL中提取matchId
    url = "https://1x2.titan007.com/oddslist/2590911.htm"
    match_id_match = re.search(r'\d+(?=\.htm$)', url)
    match_id = match_id_match.group(0) if match_id_match else None
    
    if not match_id:
        print('无法从URL中提取比赛ID')
        return None
    
    page_data = await get_all_match(url, 'odd')
    if not page_data:
        print(page_data)
        return None

    game = page_data.get('game', [])
    game_detail = page_data.get('gameDetail', [])
    
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

    # 格式化函数：将数值保留两位小数
    def format_decimal(value):
        # 先将字符串转换为浮点数，然后保留两位小数
        return f"{float(value):.2f}"

    # 解析赔率数据并进行结构调整
    def parse_odds_data(detail):
        detail_parts = detail.split('^')
        if len(detail_parts) < 2:
            print('赔率详情格式错误')
            return []

        odds_data = detail_parts[1]
        odds_array = [item for item in odds_data.split(';') if item.strip()]

        result = []
        for odds in odds_array:
            parts = odds.split('|')
            if len(parts) >= 8:
                # 前三个值：胜、平、负的赔率
                # 接下来三个值：胜、平、负的赔付比例
                # 拼接时间和年份
                full_time = f"{parts[7]}-{parts[3]}"

                result.append([
                    format_decimal(parts[0]),  # 胜赔率（保留两位小数）
                    format_decimal(parts[1]),  # 平赔率（保留两位小数）
                    format_decimal(parts[2]),  # 负赔率（保留两位小数）
                    format_decimal(parts[4]),  # 胜赔付比例（保留两位小数）
                    format_decimal(parts[5]),  # 平赔付比例（保留两位小数）
                    format_decimal(parts[6]),  # 负赔付比例（保留两位小数）
                    full_time   # 完整时间（包含年份）
                ])
        
        return result

    # 解析82和115的赔率数据
    parsed_odds_82 = parse_odds_data(target_details['82'])
    parsed_odds_115 = parse_odds_data(target_details['115'])

    # 构建结果对象，以82和115作为key，并增加oddId字段
    result = {
        'matchId': match_id,
        'oddId': {
            '82': target_odd_ids['82'],
            '115': target_odd_ids['115']
        },
        '82': parsed_odds_82,
        '115': parsed_odds_115
    }

    # 将结果存储到本地JSON文件
    # 先正常序列化，然后处理parsedOdds数组的格式，使子项内容不换行
    # 处理parsedOdds数组中的每个子项，确保子项内部没有换行
    def process_odds_array(odds_array):
        odds_strings = [json.dumps(item).replace('\n', '').replace('\s+', ' ') for item in odds_array]
        return '\n    ' + ',\n    '.join(odds_strings) + '\n  '

    # 构建完整的JSON字符串
    parsed_odds_content_82 = process_odds_array(result['82'])
    parsed_odds_content_115 = process_odds_array(result['115'])

    json_str = f'''
{{
  "matchId": "{result['matchId']}",
  "oddId": {{
    "82": "{result['oddId']['82']}",
    "115": "{result['oddId']['115']}"
  }},
  "82": [{parsed_odds_content_82}],
  "115": [{parsed_odds_content_115}]
}}
'''

    # 输出的文件名以matchId命名
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{match_id}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    print('数据解析和存储完成')
    print('结果存储在:', output_path)

# 运行异步函数
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_match_odd())