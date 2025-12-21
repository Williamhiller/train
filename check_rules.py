#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/Williamhiler/Documents/my-project/train')
from utils.feature.smart_expert_feature import SmartExpertFeatureExtractor

# 创建提取器实例
extractor = SmartExpertFeatureExtractor()

# 打印所有规则
print("=== 所有规则信息 ===")
for rule in extractor.structured_rules:
    print(f"\nID: {rule['id']}")
    print(f"规则文本: {rule['raw_text']}")
    print(f"条件类型: {[c['type'] for c in rule['conditions']]}")
    print(f"结论: {rule['conclusion']}")
    print(f"规则权重: {extractor.rule_weights.get(rule['raw_text'], '未知')}")

# 特别检查新添加的包含战绩格式的规则
print("\n=== 包含战绩格式的规则 ===")
for rule in extractor.structured_rules:
    condition_types = [c['type'] for c in rule['conditions']]
    if 'recent_record' in condition_types:
        print(f"\nID: {rule['id']}")
        print(f"规则文本: {rule['raw_text']}")
        print(f"战绩条件: {[c for c in rule['conditions'] if c['type'] == 'recent_record']}")
        print(f"规则权重: {extractor.rule_weights.get(rule['raw_text'], '未知')}")
