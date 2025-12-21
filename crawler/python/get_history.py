#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
获取比赛历史数据
"""

import asyncio
import re
import os
import json
from get_all_match import get_all_match

# 主函数
def main():
    # 从URL中提取matchId
    url = "https://zq.titan007.com/analysis/2590911cn.htm"
    match_id_match = re.search(r'\d+(?=cn\.htm$)', url)
    match_id = match_id_match.group(0) if match_id_match else None

    # 检查matchId是否成功提取
    if not match_id:
        print('无法从URL中提取比赛ID')
        return

    async def get_history():
        print('开始获取比赛数据...')
        page_data = await get_all_match(url, 'analyze')
        print('获取到的原始数据:', page_data)
        
        if not page_data:
            print('未能获取到比赛数据')
            return None
        
        away_data = page_data.get('awayData', [])
        home_data = page_data.get('homeData', [])
        history_data = page_data.get('historyData', [])
        
        # 数据解析函数：从原始数据数组中提取所需字段
        def parse_match_data(match_array):
            # 字段索引：0-比赛时间，1-联赛id，4-主队id，6-客队id，8-主队进球数，9-客队进球数，17-赛果
            # 返回数组形式，顺序：比赛时间、联赛id、主队id、客队id、主队进球数、客队进球数、赛果
            return [
                match_array[0],     # 比赛时间
                match_array[1],     # 联赛id
                match_array[4],     # 主队id
                match_array[6],     # 客队id
                match_array[8],     # 主队进球数
                match_array[9],     # 客队进球数
                match_array[12]     # 赛果
            ]

        # 解析所有数据
        parsed_data = {
            'awayData': [parse_match_data(item) for item in away_data],
            'homeData': [parse_match_data(item) for item in home_data],
            'historyData': [parse_match_data(item) for item in history_data]
        }

        # 处理数组中的每个子项，确保子项内部没有换行，格式更紧凑
        def process_data_array(data_array):
            data_strings = [json.dumps(item).replace('\n', '').replace('\s+', ' ') for item in data_array]
            return '\n    ' + ',\n    '.join(data_strings) + '\n  '

        # 构建完整的JSON字符串
        away_data_content = process_data_array(parsed_data['awayData'])
        home_data_content = process_data_array(parsed_data['homeData'])
        history_data_content = process_data_array(parsed_data['historyData'])

        json_str = f'''
{{
  "awayData": [{away_data_content}],
  "homeData": [{home_data_content}],
  "historyData": [{history_data_content}]
}}
'''

        # 保存到JSON文件
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'historyData.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        print(f'数据已保存到: {output_path}')

        return parsed_data

    # 运行异步函数
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_history())

if __name__ == "__main__":
    main()
