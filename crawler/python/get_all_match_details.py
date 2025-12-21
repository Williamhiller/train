#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
读取指定赛季目录下的文件，获取所有比赛信息
遍历获取每场比赛的详情，并保存为round-matchId名称的文件
添加延时机制以避免被识别为爬虫
"""

import asyncio
import os
import json
import random
from get_match_detail import get_match_detail

# 延时函数，随机生成延时时间，避免被识别为爬虫
async def delay(min_ms, max_ms):
    delay_time = random.randint(min_ms, max_ms)
    print(f'等待 {delay_time} 毫秒后继续...')
    await asyncio.sleep(delay_time / 1000)

# 读取指定赛季目录下的文件，获取所有比赛信息
async def read_season_data(season):
    season_dir = os.path.join(os.path.dirname(__file__), 'output', season)
    
    # 检查赛季目录是否存在
    if not os.path.exists(season_dir):
        raise FileNotFoundError(f'赛季目录 {season_dir} 不存在')
    
    # 读取赛季目录下的所有文件
    files = os.listdir(season_dir)
    
    # 过滤出JSON文件
    json_files = [file for file in files if file.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f'赛季目录 {season_dir} 下没有JSON文件')
    
    # 读取第一个JSON文件（假设每个赛季只有一个JSON文件）
    data_file = json_files[0]
    data_file_path = os.path.join(season_dir, data_file)
    
    print(f'正在读取赛季数据文件: {data_file_path}')
    
    with open(data_file_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    
    return match_data

# 为指定比赛创建详细信息目录
def create_match_detail_dir(season, round_num, match_id):
    # 创建赛季详细信息目录
    season_detail_dir = os.path.join(os.path.dirname(__file__), 'output', season, 'details')
    if not os.path.exists(season_detail_dir):
        os.makedirs(season_detail_dir)
    
    # 创建轮次目录
    round_dir = os.path.join(season_detail_dir, round_num)
    if not os.path.exists(round_dir):
        os.makedirs(round_dir)
    
    return round_dir

# 主函数，读取赛季数据并遍历获取每场比赛的详情
async def main():
    season = '2015-2016'  # 要处理的赛季
    processed_matches = 0
    failed_matches = 0
    failed_matches_list = []
    
    try:
        print(f'开始处理 {season} 赛季的所有比赛详情...')
        
        # 读取赛季数据
        match_data = await read_season_data(season)
        total_matches = len(match_data)
        
        print(f'成功读取赛季数据，共 {total_matches} 场比赛')
        
        # 遍历所有比赛，获取详细信息
        index = 0
        for match_id, match_info in match_data.items():
            index += 1
            round_num = match_info['round']
            
            print(f'\n正在处理第 {index}/{total_matches} 场比赛')
            print(f'比赛ID: {match_id}, 轮次: {round_num}')
            print(f'主队: {match_info.get("homeTeamName", "未知")}, 客队: {match_info.get("awayTeamName", "未知")}')
            
            try:
                # 创建比赛详细信息目录
                round_dir = create_match_detail_dir(season, round_num, match_id)
                
                # 生成文件名：round-matchId.json
                file_name = f'{match_id}.json'
                file_path = os.path.join(round_dir, file_name)
                
                # 检查文件是否已存在，如果存在则跳过
                if os.path.exists(file_path):
                    print(f'文件已存在，跳过抓取: {file_path}')
                    processed_matches += 1
                else:
                    # 获取比赛详情并保存到指定路径，传入赛季信息
                    detail = await get_match_detail(match_id, file_path, season)
                
                if detail:
                    processed_matches += 1
                else:
                    print('获取比赛详情失败')
                    failed_matches += 1
                    failed_matches_list.append({'matchId': match_id, 'round': round_num})

                # 如果不是最后一场比赛，添加延时
                if index < total_matches:
                    # 添加3-5秒的随机延时，避免被识别为爬虫
                    await delay(500, 2000)
            
            except Exception as error:
                print(f'处理比赛 {match_id} 时出错: {error}')
                failed_matches += 1
                failed_matches_list.append({'matchId': match_id, 'round': round_num})
                
                # 即使出错也继续处理下一场比赛，并添加延时
                # if index < total_matches:
                #     await delay(1000, 2000)  # 出错后使用较短的延时
        
        print(f'\n=== 处理完成 ===')
        print(f'总比赛数: {total_matches}')
        print(f'成功处理: {processed_matches}')
        print(f'处理失败: {failed_matches}')
        if failed_matches_list:
            print(f'\n失败的比赛记录:')
            for i, failed_match in enumerate(failed_matches_list, 1):
                print(f'{i}. 轮次: {failed_match["round"]}, 比赛ID: {failed_match["matchId"]}')
            print(f'\n失败记录总数: {len(failed_matches_list)}')
        print(f'================')
        
    except Exception as error:
        print(f'执行过程中发生错误: {error}')
        import traceback
        traceback.print_exc()

# 启动主函数
if __name__ == "__main__":
    asyncio.run(main())
