#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
获取多个赛季的比赛列表数据
"""

import asyncio
import os
import json
import random
from get_all_match import get_all_match

# 延时函数，随机生成延时时间，避免被识别为爬虫
async def delay(min_ms, max_ms):
    delay_time = random.randint(min_ms, max_ms)
    print(f'等待 {delay_time} 毫秒后继续...')
    await asyncio.sleep(delay_time / 1000)

# 处理单个赛季的数据
async def process_season(season, league_id):
    try:
        print(f'开始处理 {season} 赛季的数据...')
        url = f'http://zq.titan007.com/cn/League/{season}/{league_id}.html'
        print(f'正在访问URL: {url}')
        page_data = await get_all_match(url)
        
        # 检查pageData是否有效
        if not page_data or not isinstance(page_data, dict):
            print(f'{season} 赛季返回的数据无效')
            return 0
        
        # 构建球队字典，处理arrTeam可能不存在的情况
        team_dic = {}
        if isinstance(page_data.get('arrTeam'), list):
            for item in page_data['arrTeam']:
                if isinstance(item, list) and len(item) >= 2:
                    team_dic[item[0]] = item[1]
        
        # 创建key-value格式的数据结构，只保存比赛代码、联赛代码、主队名称、客队名称
        match_data = {}
        
        # 循环处理每个比赛，处理jh可能不存在的情况
        jh = page_data.get('jh', {})
        if isinstance(jh, dict):
            for key, round_matches in jh.items():
                if isinstance(round_matches, list):
                    for item in round_matches:
                        if isinstance(item, list) and len(item) >= 6:
                            # 获取比赛代码作为key
                            match_id = item[0]
                            
                            # 检查是否有比分，没有比分则跳过
                            score = item[6]
                            if not score or score == '' or score == '-':
                                print(f'跳过无比分的比赛: {match_id}')
                                continue
                            
                            # 解析比分并判断赛果
                            score_parts = score.split('-')
                            if len(score_parts) == 2:
                                try:
                                    home_score = int(score_parts[0])
                                    away_score = int(score_parts[1])
                                except ValueError:
                                    print(f'跳过比分格式无效的比赛: {match_id}, 比分: {score}')
                                    continue
                                
                                if home_score > away_score:
                                    match_result = 3  # 主队胜
                                elif home_score < away_score:
                                    match_result = 0  # 客队胜
                                else:
                                    match_result = 1  # 平局
                                
                                # 创建包含所需字段的对象作为value
                                match_data[match_id] = {
                                    'round': key,
                                    'matchId': item[0],  
                                    'season': season,             # 赛季信息
                                    'homeTeamName': team_dic.get(item[4], '未知'),  # 主队名称，从teamDic获取
                                    'awayTeamName': team_dic.get(item[5], '未知'),   # 客队名称，从teamDic获取
                                    'homeTeamId': item[4],
                                    'awayTeamId': item[5],
                                    'score': score, # 比分
                                    'result': match_result, # 根据比分判断的赛果
                                    'homeScore': home_score, # 主队进球数
                                    'awayScore': away_score, # 客队进球数
                                }
                            else:
                                print(f'跳过比分格式无效的比赛: {match_id}, 比分: {score}')
                                continue
        
        print(f'{season} 赛季处理完成，共获取比赛数量: {len(match_data)}')
        
        # 保存数据到本地JSON文件，使用默认联赛信息如果arrLeague不存在
        league_info = page_data.get('arrLeague', [league_id, '英格兰超级联赛'])
        save_data_to_json(match_data, league_info, season)
        
        # 返回处理的比赛数量
        return len(match_data)
        
    except Exception as e:
        print(f'处理 {season} 赛季时出错: {e}')
        # 添加更详细的错误信息
        import traceback
        print('错误堆栈:', traceback.format_exc()[:200])  # 只显示部分堆栈，避免输出过长
        return 0

# 保存数据到本地JSON文件
def save_data_to_json(data, arr_league, season):
    # 确保输出目录存在
    output_dir = os.path.join('/Users/Williamhiler/Documents/my-project/train', 'data', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建以赛季名称命名的子目录
    season_dir = os.path.join(output_dir, season)
    if not os.path.exists(season_dir):
        os.makedirs(season_dir)
    
    # 生成文件名 (移除特殊字符)
    file_name = f'{arr_league[0]}_{season}.json'
    file_path = os.path.join(season_dir, file_name)
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f'数据已成功保存至: {file_path}')

# 主函数，遍历处理多个赛季的数据
async def main():
    # 定义要处理的赛季列表，从最近的赛季开始处理，更容易测试
    seasons = [
        '2024-2025',  # 先处理最近的赛季
        '2023-2024',
        # '2022-2023',
        # '2021-2022',
        # '2020-2021',
        # '2019-2020',
        # '2018-2019',
        # '2017-2018',
        # '2016-2017',
        # '2015-2016',
    ]
    
    league_id = 36  # 英格兰超级联赛ID
    total_matches = 0
    successful_seasons = 0
    
    print('开始遍历赛季数据...')
    
    # 依次处理每个赛季
    for i, season in enumerate(seasons):
        try:
            # 处理当前赛季
            match_count = await process_season(season, league_id)
            
            if match_count > 0:
                successful_seasons += 1
                total_matches += match_count
                print(f'✅ {season} 赛季成功处理')
            else:
                print(f'⚠️  {season} 赛季未获取到数据或处理失败')
            
            # 如果不是最后一个赛季，添加延时
            if i < len(seasons) - 1:
                # 添加10-20秒的随机延时（减少测试时间），避免被识别为爬虫
                await delay(10000, 20000)
        except Exception as e:
            print(f'❌ 处理 {season} 赛季时发生异常: {e}')
            # 即使出错也继续处理下一个赛季
            if i < len(seasons) - 1:
                await delay(5000, 10000)  # 出错后使用较短的延时
    
    print(f'\n=== 处理完成 ===')
    print(f'成功处理的赛季数量: {successful_seasons}/{len(seasons)}')
    print(f'总计获取比赛数量: {total_matches}')
    print(f'================')

# 启动主函数
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
