#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备脚本：生成预处理数据和上下文
"""

import os
import sys
import yaml
import json
import pandas as pd
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing.match_data_processor import MatchDataProcessor
from utils.data_processing.context_extractor import ContextExtractor


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_processed_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """生成预处理数据，按赛季分批保存"""
    print("=" * 80)
    print("步骤1: 生成预处理数据（按赛季分批）")
    print("=" * 80)
    
    raw_data_path = config["data"]["raw_data_path"]
    processed_data_path = config["data"]["processed_data_path"]
    
    print(f"原始数据路径: {raw_data_path}")
    print(f"预处理数据保存路径: {processed_data_path}")
    
    processor = MatchDataProcessor(config)
    
    print("开始处理比赛数据...")
    season_dfs = processor.process_data(raw_data_path, save_by_season=True)
    
    print(f"处理完成，共 {len(season_dfs)} 个赛季")
    
    os.makedirs(processed_data_path, exist_ok=True)
    
    for season, df in season_dfs.items():
        csv_path = os.path.join(processed_data_path, f"processed_matches_{season}.csv")
        json_path = os.path.join(processed_data_path, f"processed_matches_{season}.json")
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"赛季 {season} 数据已保存到: {csv_path}")
        
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"赛季 {season} 数据已保存到: {json_path}")
    
    total_matches = sum(len(df) for df in season_dfs.values())
    print(f"总计 {total_matches} 条记录")
    
    return season_dfs


def generate_contexts(config: Dict) -> Dict[str, pd.DataFrame]:
    """生成上下文数据，按赛季分批保存"""
    print("\n" + "=" * 80)
    print("步骤2: 生成上下文数据（按赛季分批）")
    print("=" * 80)
    
    raw_data_path = config["data"]["raw_data_path"]
    
    print(f"原始数据路径: {raw_data_path}")
    
    extractor = ContextExtractor()
    
    print("开始生成上下文（不包含赛果）...")
    season_contexts = extractor.generate_contexts_from_raw(
        raw_data_path=raw_data_path,
        include_result=False,
        context_type='knowledge_matching',
        save_to_file=True,
        filename_prefix="contexts_knowledge_matching"
    )
    
    print(f"上下文生成完成，共 {len(season_contexts)} 个赛季")
    
    total_contexts = sum(len(df) for df in season_contexts.values())
    print(f"总计 {total_contexts} 条记录")
    
    return season_contexts


def main():
    """主函数"""
    config_path = "configs/v5_config.yaml"
    
    print("开始数据准备流程...")
    print(f"配置文件: {config_path}")
    
    config = load_config(config_path)
    
    try:
        processed_df = generate_processed_data(config)
        
        # 暂时跳过scaler初始化，因为上下文生成不需要
        # print("\nInitializing scaler with all data...")
        # from utils.data_processing.match_data_processor import MatchDataProcessor
        # processor = MatchDataProcessor(config)
        # all_data = pd.concat(processed_df.values(), ignore_index=True)
        # processor.initialize_scaler(all_data)
        
        contexts_dict = generate_contexts(config)
        
        print("\n" + "=" * 80)
        print("数据准备完成!")
        print("=" * 80)
        print(f"预处理数据: {len(processed_df)} 条记录")
        print(f"上下文数据: {len(contexts_dict)} 条记录")
        print("\n现在可以开始训练模型了！")
        
    except Exception as e:
        print(f"\n数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
