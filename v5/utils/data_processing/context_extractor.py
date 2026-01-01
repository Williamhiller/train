#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上下文提取器，用于批量生成原始数据的上下文，供后续训练使用
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .context_generator import ContextGenerator


class ContextExtractor:
    """上下文提取器类，用于批量生成原始数据的上下文"""
    
    def __init__(self):
        """初始化上下文提取器"""
        # 集成ContextGenerator（与测试时使用相同的生成器）
        self.context_generator = ContextGenerator()
        self.save_dir = "contexts"
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
    
    def extract_context(self, match_data: Dict, include_result: bool = False, context_type: str = 'knowledge_matching') -> str:
        """提取单个比赛的上下文
        
        Args:
            match_data: 比赛数据字典
            include_result: 是否包含赛果信息
            context_type: 上下文类型，可选值：knowledge_matching, prediction
            
        Returns:
            生成的上下文字符串
        """
        try:
            # 确保match_data是字典类型
            if isinstance(match_data, dict):
                # 使用ContextGenerator生成上下文，确保与测试时一致
                context = self.context_generator.generate_context(
                    match_data=match_data,
                    context_type=context_type,
                    include_result=include_result
                )
                return context
            else:
                print(f"生成上下文失败: match_data不是字典类型，而是{type(match_data)}")
                return ""
        except Exception as e:
            print(f"生成上下文失败: {e}")
            return ""
    
    def extract_contexts_batch(self, match_data_list: List[Dict], 
                             include_result: bool = False, 
                             context_type: str = 'knowledge_matching',
                             show_progress: bool = True) -> List[Tuple[str, str]]:
        """批量提取多个比赛的上下文
        
        Args:
            match_data_list: 比赛数据字典列表
            include_result: 是否包含赛果信息
            context_type: 上下文类型
            show_progress: 是否显示进度条
            
        Returns:
            包含比赛ID和上下文的元组列表
        """
        results = []
        
        # 使用进度条
        iterator = tqdm(match_data_list, desc="生成上下文") if show_progress else match_data_list
        
        for match_data in iterator:
            # 确保match_data是字典类型
            if isinstance(match_data, dict):
                match_id = match_data.get('matchId', match_data.get('match_id', 'unknown'))
                context = self.extract_context(match_data, include_result, context_type)
                results.append((str(match_id), context))
            else:
                # 如果match_data不是字典，尝试获取match_id
                try:
                    # 假设match_data可能是字符串格式的match_id
                    match_id = str(match_data)
                    context = ""
                    results.append((match_id, context))
                except Exception as e:
                    print(f"处理比赛数据失败: {e}")
                    results.append(("unknown", ""))
        
        return results
    
    def load_raw_data(self, data_path: str) -> List[Dict]:
        """加载原始比赛数据，包括赛季根目录下的json文件信息
        
        Args:
            data_path: 原始数据路径
            
        Returns:
            比赛数据字典列表
        """
        all_matches = []
        season_matches_map = {}  # 用于存储赛季文件中的比赛信息，match_id -> match_info
        
        print(f"正在遍历数据路径: {data_path}")
        print(f"路径是否存在: {os.path.exists(data_path)}")
        
        # 第一步：加载所有赛季根目录下的json文件，获取比赛基本信息和赛果
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            
            # 检查是否是赛季目录
            if os.path.isdir(item_path) and item.startswith('20'):
                season_dir = item
                season_path = item_path
                print(f"处理赛季目录: {season_path}")
                
                # 加载赛季根目录下的json文件（如36_2017-2018.json）
                season_files = [f for f in os.listdir(season_path) if f.endswith('.json') and not 'details' in f]
                for season_file in season_files:
                    season_file_path = os.path.join(season_path, season_file)
                    try:
                        with open(season_file_path, 'r', encoding='utf-8') as f:
                            season_data = json.load(f)
                        
                        print(f"加载赛季文件: {season_file_path}")
                        
                        # 赛季文件可能是字典或列表
                        if isinstance(season_data, dict):
                            # 字典格式：match_id -> match_info
                            for match_id, match_info in season_data.items():
                                # 确保match_id是字符串
                                match_id_str = str(match_id)
                                season_matches_map[match_id_str] = match_info
                        elif isinstance(season_data, list):
                            # 列表格式：直接是match_info列表
                            for match_info in season_data:
                                match_id = str(match_info.get('matchId', match_info.get('match_id', '')))
                                if match_id:
                                    season_matches_map[match_id] = match_info
                    except Exception as e:
                        print(f"加载赛季文件失败 {season_file_path}: {e}")
        
        print(f"从赛季文件加载了 {len(season_matches_map)} 个比赛基本信息")
        
        # 第二步：加载details目录下的比赛详细信息
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            
            # 检查是否是赛季目录
            if os.path.isdir(item_path) and item.startswith('20'):
                season_dir = item
                season_path = item_path
                
                # 检查是否有details子目录
                details_path = os.path.join(season_path, "details")
                if os.path.isdir(details_path):
                    print(f"处理details目录: {details_path}")
                    
                    # 遍历details目录下的所有轮次目录
                    for round_dir in os.listdir(details_path):
                        round_path = os.path.join(details_path, round_dir)
                        if os.path.isdir(round_path):
                            print(f"处理轮次目录: {round_path}")
                            
                            # 遍历轮次目录下的所有比赛文件
                            for match_file in os.listdir(round_path):
                                if match_file.endswith('.json'):
                                    file_path = os.path.join(round_path, match_file)
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            match_data = json.load(f)
                                        
                                        # 从文件名提取match_id
                                        match_id = match_file.replace('.json', '')
                                        
                                        # 如果赛季文件中有这个比赛的信息，合并进来
                                        if match_id in season_matches_map:
                                            # 合并赛季文件信息到比赛数据中
                                            season_info = season_matches_map[match_id]
                                            # 保留原始比赛数据中的信息，用赛季文件信息补充
                                            match_data = {**season_info, **match_data}
                                            print(f"合并赛季信息到比赛 {match_id}")
                                        
                                        # 添加基本信息
                                        match_data['matchId'] = match_id
                                        match_data['season'] = season_dir
                                        match_data['file_path'] = file_path
                                        
                                        # 确保主客队名称存在
                                        if 'homeTeam' not in match_data and 'homeTeamName' in match_data:
                                            match_data['homeTeam'] = match_data['homeTeamName']
                                        if 'awayTeam' not in match_data and 'awayTeamName' in match_data:
                                            match_data['awayTeam'] = match_data['awayTeamName']
                                        
                                        all_matches.append(match_data)
                                    except Exception as e:
                                        print(f"加载文件失败 {file_path}: {e}")
        
        print(f"总共加载了 {len(all_matches)} 个比赛数据")
        return all_matches
    
    def generate_contexts_from_raw(self, raw_data_path: str,
                                 include_result: bool = False,
                                 context_type: str = 'knowledge_matching',
                                 save_to_file: bool = True,
                                 filename_prefix: str = "contexts") -> Dict[str, pd.DataFrame]:
        """从原始数据生成上下文，按赛季分批保存
        
        Args:
            raw_data_path: 原始数据路径
            include_result: 是否包含赛果信息
            context_type: 上下文类型
            save_to_file: 是否保存到文件
            filename_prefix: 文件名前缀
            
        Returns:
            赛季->DataFrame的字典
        """
        # 1. 加载原始数据
        print(f"正在加载原始数据: {raw_data_path}")
        match_data_list = self.load_raw_data(raw_data_path)
        print(f"成功加载 {len(match_data_list)} 个比赛数据")
        
        # 2. 按赛季分组生成上下文
        season_contexts = {}
        for season in set(match.get('season', '') for match in match_data_list):
            if not season:
                continue
            
            print(f"正在生成赛季 {season} 的上下文...")
            season_matches = [m for m in match_data_list if m.get('season', '') == season]
            
            # 批量生成该赛季的上下文
            contexts = self.extract_contexts_batch(
                match_data_list=season_matches,
                include_result=include_result,
                context_type=context_type
            )
            
            # 构建DataFrame
            data = []
            for match_id, context in contexts:
                data.append({
                    'match_id': match_id,
                    'season': season,
                    'context': context,
                    'include_result': include_result
                })
            
            df = pd.DataFrame(data)
            season_contexts[season] = df
            print(f"赛季 {season}: 生成 {len(df)} 个上下文")
        
        # 3. 保存到文件
        if save_to_file:
            result_suffix = "_with_result" if include_result else "_without_result"
            
            for season, df in season_contexts.items():
                filename = f"{filename_prefix}_{season}_{context_type}{result_suffix}.csv"
                file_path = os.path.join(self.save_dir, filename)
                
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"赛季 {season} 上下文已保存到: {file_path}")
            
            # 同时保存一个合并的JSON文件，方便训练时使用
            all_contexts = {}
            for season, df in season_contexts.items():
                for _, row in df.iterrows():
                    all_contexts[str(row['match_id'])] = row['context']
            
            json_filename = f"{filename_prefix}_{context_type}{result_suffix}.json"
            json_file_path = os.path.join(self.save_dir, json_filename)
            
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_contexts, f, ensure_ascii=False, indent=2)
            
            print(f"所有上下文已保存到JSON文件: {json_file_path}")
        
        return season_contexts
    
    def get_context_statistics(self, contexts_df: pd.DataFrame) -> Dict:
        """获取上下文统计信息
        
        Args:
            contexts_df: 包含上下文的DataFrame
            
        Returns:
            统计信息字典
        """
        if contexts_df.empty:
            return {
                'total_contexts': 0,
                'average_length': 0,
                'min_length': 0,
                'max_length': 0,
                'include_result': False,
                'season_distribution': {}
            }
        
        # 检查必要列是否存在
        has_context = 'context' in contexts_df.columns
        has_include_result = 'include_result' in contexts_df.columns
        has_season = 'season' in contexts_df.columns
        
        stats = {
            'total_contexts': len(contexts_df),
            'average_length': contexts_df['context'].str.len().mean() if has_context else 0,
            'min_length': contexts_df['context'].str.len().min() if has_context else 0,
            'max_length': contexts_df['context'].str.len().max() if has_context else 0,
            'include_result': contexts_df['include_result'].iloc[0] if has_include_result and len(contexts_df) > 0 else False,
            'season_distribution': contexts_df['season'].value_counts().to_dict() if has_season else {}
        }
        
        return stats
    
    def save_contexts_to_json(self, contexts: List[Tuple[str, str]], filename: str = "contexts.json") -> str:
        """保存上下文到JSON文件
        
        Args:
            contexts: 包含比赛ID和上下文的元组列表
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        file_path = os.path.join(self.save_dir, filename)
        
        # 转换为字典格式
        context_dict = {match_id: context for match_id, context in contexts}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(context_dict, f, ensure_ascii=False, indent=2)
        
        print(f"上下文已保存到JSON文件: {file_path}")
        return file_path
    
    def load_contexts_from_file(self, file_path: str) -> pd.DataFrame:
        """从文件加载上下文
        
        Args:
            file_path: 上下文文件路径
            
        Returns:
            包含上下文的DataFrame
        """
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                context_dict = json.load(f)
            
            data = [{'match_id': match_id, 'context': context} for match_id, context in context_dict.items()]
            return pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def validate_context_consistency(self, context1: str, context2: str) -> float:
        """验证两个上下文的一致性
        
        Args:
            context1: 第一个上下文
            context2: 第二个上下文
            
        Returns:
            相似度分数 (0-1)
        """
        # 简单的相似度计算，比较上下文包含的关键字段
        core_fields = ["比赛ID", "比赛球队", "威廉初赔", "威廉终赔", "立博初赔", "立博终赔", 
                      "主队近期状态", "客队近期状态", "主队近6场", "客队近6场", "对战历史"]
        
        context1_has = [field in context1 for field in core_fields]
        context2_has = [field in context2 for field in core_fields]
        
        # 计算匹配的字段数量
        matches = sum(h1 and h2 for h1, h2 in zip(context1_has, context2_has))
        return matches / len(core_fields)
