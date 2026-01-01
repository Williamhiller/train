#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家知识提取器
利用Qwen的语义理解能力，从专家文本中提取结构化的预测规则
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


@dataclass
class ExpertRule:
    """专家规则数据结构"""
    rule_id: str
    title: str
    category: str  # 规则类别：赔率分析、开盘思路、心理分析等
    content: str
    conditions: List[str]  # 适用条件
    key_points: List[str]  # 关键要点
    examples: List[str]  # 示例
    embedding: Optional[np.ndarray] = None  # 用于语义匹配的嵌入向量


class ExpertKnowledgeExtractor:
    """专家知识提取器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rules: List[ExpertRule] = []
        
        # 初始化Qwen模型用于语义理解
        self.model_path = config.get("qwen", {}).get("model_path")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载Qwen模型用于专家知识提取...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"✅ Qwen模型加载成功")
    
    def extract_rules_from_expert_data(self, expert_data_path: str) -> List[ExpertRule]:
        """从专家数据中提取规则
        
        Args:
            expert_data_path: 专家数据文件路径
            
        Returns:
            提取的规则列表
        """
        import time
        start_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}] 正在从 {expert_data_path} 提取专家规则...")
        
        # 加载专家数据
        with open(expert_data_path, 'r', encoding='utf-8') as f:
            expert_data = json.load(f)
        
        total_items = len(expert_data)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] 加载了 {total_items} 条专家数据")
        
        rules = []
        
        for idx, item in enumerate(expert_data, 1):
            # 每10条打印一次进度
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] 处理进度: {idx}/{total_items} ({idx/total_items*100:.1f}%), 已用时间: {elapsed:.1f}s")
            
            # 使用Qwen理解专家文本并提取规则
            rule = self._extract_rule_from_text(item)
            if rule:
                rules.append(rule)
        
        elapsed_time = time.time() - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] ✅ 提取完成，共 {len(rules)} 条专家规则，总用时: {elapsed_time:.1f}s")
        self.rules = rules
        return rules
    
    def _extract_rule_from_text(self, expert_item: Dict) -> Optional[ExpertRule]:
        """从单个专家文本中提取规则
        
        Args:
            expert_item: 专家数据项
            
        Returns:
            提取的规则
        """
        prompt = expert_item.get("prompt", "")
        response = expert_item.get("response", "")
        
        # 构建提取提示
        extraction_prompt = f"""
请从以下专家分析中提取足球比赛预测的关键规则，以JSON格式返回：

专家分析：
{response}

请提取以下信息：
1. 规则标题（简短描述）
2. 规则类别（如：赔率分析、开盘思路、心理分析、平衡法则等）
3. 规则内容（详细说明）
4. 适用条件（什么情况下适用）
5. 关键要点（3-5个要点）
6. 示例（如果有）

返回格式：
{{
  "title": "规则标题",
  "category": "规则类别",
  "content": "规则内容",
  "conditions": ["条件1", "条件2"],
  "key_points": ["要点1", "要点2", "要点3"],
  "examples": ["示例1"]
}}
"""
        
        # 使用Qwen生成结构化规则
        try:
            inputs = self.tokenizer(
                extraction_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 这里简化处理，实际应该使用生成模型
                # 暂时使用规则匹配的方式
            
            # 使用规则匹配方式提取
            rule = self._rule_based_extraction(prompt, response)
            
            # 生成嵌入向量用于语义匹配
            if rule:
                rule.embedding = self._generate_embedding(rule.content)
            
            return rule
            
        except Exception as e:
            print(f"❌ 提取规则失败: {e}")
            return None
    
    def _rule_based_extraction(self, prompt: str, response: str) -> Optional[ExpertRule]:
        """基于规则的方式提取专家知识
        
        Args:
            prompt: 提示文本
            response: 响应文本
            
        Returns:
            提取的规则
        """
        # 简化的规则提取逻辑
        text = prompt + "\n" + response
        
        # 尝试识别规则类别
        category = self._identify_category(text)
        
        # 提取关键内容
        content = self._extract_key_content(text)
        
        # 提取关键要点
        key_points = self._extract_key_points(text)
        
        # 提取适用条件
        conditions = self._extract_conditions(text)
        
        if not content or len(content) < 50:
            return None
        
        rule_id = f"rule_{len(self.rules) + 1}"
        title = content[:50] + "..."
        
        return ExpertRule(
            rule_id=rule_id,
            title=title,
            category=category,
            content=content,
            conditions=conditions,
            key_points=key_points,
            examples=[]
        )
    
    def _identify_category(self, text: str) -> str:
        """识别规则类别
        
        Args:
            text: 文本
            
        Returns:
            规则类别
        """
        categories = {
            "赔率分析": ["赔率", "欧赔", "盘面", "主陪", "副陪", "渣陪"],
            "开盘思路": ["开盘", "出3", "出1", "出0", "开盘思维"],
            "平衡法则": ["平衡", "失衡", "均衡", "分流"],
            "心理分析": ["心理", "诱导", "阻力", "拉力", "迷惑"],
            "实战技巧": ["复盘", "悟", "借", "协作"]
        }
        
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[category] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "其他"
    
    def _extract_key_content(self, text: str) -> str:
        """提取关键内容
        
        Args:
            text: 文本
            
        Returns:
            关键内容
        """
        # 简化处理：提取包含"法则"、"思路"、"手法"等关键词的段落
        sentences = text.split('\n')
        key_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in ["法则", "思路", "手法", "思维", "原则"]):
                key_sentences.append(sentence.strip())
        
        return '\n'.join(key_sentences[:3]) if key_sentences else text[:500]
    
    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键要点
        
        Args:
            text: 文本
            
        Returns:
            关键要点列表
        """
        points = []
        sentences = text.split('\n')
        
        for sentence in sentences:
            # 提取以数字开头的要点
            if sentence.strip() and sentence.strip()[0].isdigit():
                point = sentence.strip()
                if len(point) > 10 and len(point) < 200:
                    points.append(point)
        
        return points[:5] if points else []
    
    def _extract_conditions(self, text: str) -> List[str]:
        """提取适用条件
        
        Args:
            text: 文本
            
        Returns:
            适用条件列表
        """
        conditions = []
        sentences = text.split('\n')
        
        for sentence in sentences:
            # 提取包含"如果"、"当"、"则"等条件的句子
            if any(keyword in sentence for keyword in ["如果", "当", "则", "时", "以下"]):
                condition = sentence.strip()
                if len(condition) > 10 and len(condition) < 150:
                    conditions.append(condition)
        
        return conditions[:3] if conditions else []
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入向量
        
        Args:
            text: 文本
            
        Returns:
            嵌入向量
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的输出
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # 归一化
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            print(f"❌ 生成嵌入失败: {e}")
            return np.zeros(768)
    
    def save_rules(self, output_path: str):
        """保存提取的规则
        
        Args:
            output_path: 输出路径
        """
        rules_data = []
        for rule in self.rules:
            rule_dict = {
                "rule_id": rule.rule_id,
                "title": rule.title,
                "category": rule.category,
                "content": rule.content,
                "conditions": rule.conditions,
                "key_points": rule.key_points,
                "examples": rule.examples
            }
            rules_data.append(rule_dict)
        
        # 保存规则
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 规则已保存到 {output_path}")
    
    def load_rules(self, rules_path: str) -> List[ExpertRule]:
        """加载规则
        
        Args:
            rules_path: 规则文件路径
            
        Returns:
            规则列表
        """
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        self.rules = []
        for rule_dict in rules_data:
            rule = ExpertRule(
                rule_id=rule_dict["rule_id"],
                title=rule_dict["title"],
                category=rule_dict["category"],
                content=rule_dict["content"],
                conditions=rule_dict["conditions"],
                key_points=rule_dict["key_points"],
                examples=rule_dict.get("examples", [])
            )
            
            # 生成嵌入
            rule.embedding = self._generate_embedding(rule.content)
            self.rules.append(rule)
        
        print(f"✅ 加载了 {len(self.rules)} 条专家规则")
        return self.rules


if __name__ == "__main__":
    import sys
    import yaml
    
    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/v5_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建提取器
    extractor = ExpertKnowledgeExtractor(config)
    
    # 提取规则
    expert_data_path = "data/expert_data/expert_training_data.json"
    rules = extractor.extract_rules_from_expert_data(expert_data_path)
    
    # 保存规则
    output_path = "data/expert_knowledge/expert_rules.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extractor.save_rules(output_path)
