#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能专家特征提取器测试脚本
"""

import os
import sys
import json
import logging
from utils.feature.smart_expert_feature import SmartExpertFeatureExtractor
from utils.feature.expert_feature import ExpertFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartExpertFeatureTester:
    """
    智能专家特征提取器测试类
    """
    
    def __init__(self):
        """
        初始化测试类
        """
        self.base_path = "/Users/Williamhiler/Documents/my-project/train"
        self.test_data_path = os.path.join(self.base_path, "test_data")
        self.odds_data_path = os.path.join(self.base_path, "train-data/odds")
        self.team_state_path = os.path.join(self.base_path, "train-data/team_state")
        
        # 创建数据目录（如果不存在）
        os.makedirs(self.test_data_path, exist_ok=True)
        
        # 加载测试数据
        self.test_matches = self._load_test_matches()
        
        # 初始化提取器
        self.smart_extractor = SmartExpertFeatureExtractor()
        self.original_extractor = ExpertFeatureExtractor()
        
        logger.info(f"智能专家特征提取器测试初始化完成")
        logger.info(f"已加载 {len(self.test_matches)} 个测试比赛")
    
    def _load_test_matches(self):
        """
        加载测试比赛数据
        """
        test_matches = []
        
        # 查找最新的赔率和球队状态数据文件
        odds_files = sorted([f for f in os.listdir(self.odds_data_path) if f.endswith('.json')], reverse=True)
        team_state_files = sorted([f for f in os.listdir(self.team_state_path) if f.endswith('.json')], reverse=True)
        
        if not odds_files or not team_state_files:
            logger.error("未找到赔率或球队状态数据文件")
            return []
        
        # 加载最新的赔率数据
        latest_odds_file = os.path.join(self.odds_data_path, odds_files[0])
        latest_team_state_file = os.path.join(self.team_state_path, team_state_files[0])
        
        try:
            with open(latest_odds_file, 'r', encoding='utf-8') as f:
                odds_data = json.load(f)
            
            with open(latest_team_state_file, 'r', encoding='utf-8') as f:
                team_state_data = json.load(f)
            
            # 解析赔率数据（matches数组形式）
            if isinstance(odds_data, dict) and 'matches' in odds_data:
                odds_matches = odds_data['matches']
                
                # 创建赔率数据的映射（match_id -> 赔率数据）
                odds_data_map = {}
                for match in odds_matches:
                    if 'match_id' in match:
                        odds_data_map[match['match_id']] = match
                
                # 获取共同的比赛ID
                odds_match_ids = set(odds_data_map.keys())
                team_state_match_ids = set(team_state_data.keys())
                common_match_ids = odds_match_ids.intersection(team_state_match_ids)
                
                logger.info(f"在最新数据文件中找到 {len(common_match_ids)} 个共同比赛")
                
                # 选择前10个比赛作为测试数据
                for match_id in list(common_match_ids)[:10]:
                    match_data = {
                        'match_id': match_id,
                        'odds_data': odds_data_map[match_id],
                        'team_state_data': team_state_data[match_id]
                    }
                    test_matches.append(match_data)
            else:
                logger.error("赔率数据格式不正确")
            
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
        
        return test_matches
    
    def test_feature_extraction(self):
        """
        测试特征提取功能
        """
        logger.info("开始测试特征提取功能...")
        
        results = []
        
        for i, match_data in enumerate(self.test_matches, 1):
            match_id = match_data['match_id']
            odds_data = match_data['odds_data']
            team_state_data = match_data['team_state_data']
            
            logger.info(f"测试比赛 {i}/{len(self.test_matches)}: {match_id}")
            
            try:
                # 使用原始提取器提取特征
                original_features = self.original_extractor.extract_expert_features(match_id, odds_data, team_state_data)
                
                # 使用智能提取器提取特征
                smart_features = self.smart_extractor.extract_expert_features(match_id, odds_data, team_state_data)
                
                # 比较结果
                comparison = self._compare_features(original_features, smart_features)
                
                # 保存结果
                result = {
                    'match_id': match_id,
                    'original_features': original_features,
                    'smart_features': smart_features,
                    'comparison': comparison
                }
                results.append(result)
                
                logger.info(f"比赛 {match_id} 特征提取完成")
                logger.info(f"智能特征数量: {len(smart_features)}，原始特征数量: {len(original_features)}")
                logger.info(f"新增智能特征: {comparison['new_features']}")
                
            except Exception as e:
                logger.error(f"处理比赛 {match_id} 时出错: {e}")
        
        # 保存测试结果
        self._save_test_results(results)
        
        logger.info("特征提取功能测试完成")
        return results
    
    def _compare_features(self, original_features, smart_features):
        """
        比较原始特征和智能特征
        """
        original_keys = set(original_features.keys())
        smart_keys = set(smart_features.keys())
        
        # 共同特征
        common_features = original_keys.intersection(smart_keys)
        
        # 新特征
        new_features = smart_keys - original_keys
        
        # 检查共同特征值的差异
        feature_diffs = {}
        for key in common_features:
            if isinstance(original_features[key], (int, float)) and isinstance(smart_features[key], (int, float)):
                diff = abs(original_features[key] - smart_features[key])
                feature_diffs[key] = diff
        
        return {
            'common_features': list(common_features),
            'new_features': list(new_features),
            'feature_diffs': feature_diffs
        }
    
    def _save_test_results(self, results):
        """
        保存测试结果
        """
        results_path = os.path.join(self.test_data_path, "smart_expert_feature_test_results.json")
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            logger.info(f"测试结果已保存到 {results_path}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    def test_rule_extraction(self):
        """
        测试规则提取功能
        """
        logger.info("开始测试规则提取功能...")
        
        # 获取提取的规则
        extracted_rules = self.smart_extractor.extracted_rules
        rule_weights = self.smart_extractor.rule_weights
        
        logger.info(f"已提取 {len(extracted_rules)} 条专家规则")
        logger.info(f"规则权重数量: {len(rule_weights)}")
        
        # 保存规则分析结果
        rules_analysis = {
            'total_rules': len(extracted_rules),
            'rules_with_weights': len(rule_weights),
            'rules_sample': extracted_rules[:10],
            'weights_sample': dict(list(rule_weights.items())[:10])
        }
        
        rules_path = os.path.join(self.test_data_path, "expert_rules_analysis.json")
        
        try:
            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(rules_analysis, f, ensure_ascii=False, indent=4)
            
            logger.info(f"规则分析结果已保存到 {rules_path}")
        except Exception as e:
            logger.error(f"保存规则分析结果失败: {e}")
        
        logger.info("规则提取功能测试完成")
        return rules_analysis
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        logger.info("开始运行所有智能专家特征提取器测试...")
        
        # 测试特征提取
        feature_results = self.test_feature_extraction()
        
        # 测试规则提取
        rule_results = self.test_rule_extraction()
        
        # 汇总测试结果
        summary = {
            'test_matches_count': len(self.test_matches),
            'successful_extractions': len(feature_results),
            'extracted_rules_count': rule_results['total_rules'],
            'rules_with_weights_count': rule_results['rules_with_weights']
        }
        
        logger.info("所有测试完成")
        logger.info(f"测试总结: {json.dumps(summary, ensure_ascii=False, indent=2)}")
        
        return {
            'feature_results': feature_results,
            'rule_results': rule_results,
            'summary': summary
        }

def main():
    """
    主函数
    """
    try:
        tester = SmartExpertFeatureTester()
        results = tester.run_all_tests()
        
        logger.info("智能专家特征提取器测试成功完成")
        logger.info(f"测试匹配数量: {results['summary']['test_matches_count']}")
        logger.info(f"成功提取特征的匹配数量: {results['summary']['successful_extractions']}")
        logger.info(f"提取的规则数量: {results['summary']['extracted_rules_count']}")
        logger.info(f"有权重的规则数量: {results['summary']['rules_with_weights_count']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())