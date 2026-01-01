#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于Qwen大模型的智能专家知识匹配器
利用语义理解能力实现精准的知识匹配
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# 导入最新的上下文生成器
from v5.utils.data_processing.context_generator import ContextGenerator


class QwenKnowledgeMatcher:
    """基于Qwen的智能知识匹配器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.knowledge_base = None
        self.embeddings = None
        self.knowledge_units = []
        
        # 加载配置
        self.llm_config = config.get('expert_analysis_llm', {})
        self.force_local = self.llm_config.get('force_local', True)  # 默认为True，强制使用本地模型
        self.local_model_path = self.llm_config.get('model_name', config.get('qwen', {}).get('model_path', '/Users/Williamhiler/Documents/my-project/train/models/cache'))  # 使用配置中的模型路径
        
        # 加载知识库
        self.load_knowledge_base()
        
        # 初始化上下文生成器
        self.context_generator = ContextGenerator()
        
        # 初始化Qwen模型和分词器
        self.model, self.tokenizer = self._initialize_qwen_model()
        
        # 生成知识库嵌入
        self.generate_knowledge_embeddings()
    
    
    def _initialize_qwen_model(self):
        """初始化Qwen模型和分词器"""
        print("正在加载Qwen模型...")
        
        try:
            if self.force_local:
                print(f"🔒 强制使用本地模型")
                print(f"   本地模型路径: {self.local_model_path}")
                
                # 检查本地模型是否存在
                if not os.path.exists(self.local_model_path):
                    raise FileNotFoundError(f"本地模型路径不存在: {self.local_model_path}")
                
                # 强制从本地加载，不尝试下载
                tokenizer = AutoTokenizer.from_pretrained(
                    self.local_model_path,
                    local_files_only=True  # 强制使用本地文件
                )
                model = AutoModel.from_pretrained(
                    self.local_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    local_files_only=True  # 强制使用本地文件
                )
            else:
                # 正常模式，先尝试本地，再尝试下载
                local_model_path = "/Users/Williamhiler/Documents/my-project/train/v5/models/cache"
                
                if os.path.exists(local_model_path):
                    print(f"使用本地模型: {local_model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                    model = AutoModel.from_pretrained(
                        local_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    print(f"本地模型不存在，尝试使用模型名称...")
                    model_name = "Qwen/Qwen2-0.5B"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            
            print("✅ Qwen模型加载成功")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载Qwen模型失败: {e}")
            print(f"   本地模型路径: {self.local_model_path}")
            print(f"   强制本地模式: {self.force_local}")
            raise
    
    
    def load_knowledge_base(self):
        """加载知识库"""
        knowledge_base_path = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json"
        
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            
            self.knowledge_units = self.knowledge_base["knowledge_units"]
            print(f"✅ 成功加载知识库: {len(self.knowledge_units)} 个知识单元")
        except Exception as e:
            print(f"❌ 加载知识库失败: {e}")
            self.knowledge_base = None
    
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        try:
            # 分词
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 移至GPU
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用最后一层的CLS token作为嵌入
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # 归一化
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"❌ 生成嵌入失败: {e}")
            # 返回全零向量作为 fallback
            return np.zeros(512)
    
    
    def generate_knowledge_embeddings(self):
        """生成知识库中所有知识单元的嵌入"""
        # 先尝试加载预生成的嵌入
        if self._load_precomputed_embeddings():
            return
        
        print("正在生成知识库嵌入...")
        
        if not self.knowledge_units:
            print("❌ 知识库为空，无法生成嵌入")
            return
        
        try:
            # 生成所有知识单元的嵌入
            self.embeddings = []
            for i, unit in enumerate(self.knowledge_units):
                # 构建知识单元的文本表示
                knowledge_text = self._construct_knowledge_text(unit)
                # 生成嵌入
                embedding = self.generate_embedding(knowledge_text)
                self.embeddings.append(embedding)
                
                # 打印进度
                if (i + 1) % 100 == 0:
                    print(f"已生成 {i + 1}/{len(self.knowledge_units)} 个知识单元嵌入")
            
            # 转换为numpy数组
            self.embeddings = np.array(self.embeddings)
            print(f"✅ 完成生成知识库嵌入: 形状 {self.embeddings.shape}")
        except Exception as e:
            print(f"❌ 生成知识库嵌入失败: {e}")
            self.embeddings = None
    
    
    def _load_precomputed_embeddings(self):
        """加载预计算的嵌入向量"""
        embedding_cache_dir = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/embeddings"
        embedding_file = os.path.join(embedding_cache_dir, "knowledge_embeddings.npy")
        config_file = os.path.join(embedding_cache_dir, "embedding_config.json")
        
        if os.path.exists(embedding_file) and os.path.exists(config_file):
            print("正在加载预计算的嵌入向量...")
            try:
                # 加载嵌入向量
                self.embeddings = np.load(embedding_file)
                
                # 加载配置
                with open(config_file, 'r', encoding='utf-8') as f:
                    embedding_config = json.load(f)
                
                print(f"✅ 成功加载预计算嵌入: 形状 {self.embeddings.shape}")
                print(f"   模型名称: {embedding_config.get('model_name')}")
                print(f"   生成时间: {embedding_config.get('generated_timestamp')}")
                
                # 验证嵌入数量与知识单元数量是否匹配
                if len(self.embeddings) == len(self.knowledge_units):
                    return True
                else:
                    print(f"❌ 嵌入数量不匹配: {len(self.embeddings)} != {len(self.knowledge_units)}")
                    return False
                    
            except Exception as e:
                print(f"❌ 加载预计算嵌入失败: {e}")
                return False
        
        print("未找到预计算的嵌入向量，将生成新的嵌入")
        return False
    
    
    def _construct_knowledge_text(self, knowledge_unit: Dict) -> str:
        """构建知识单元的文本表示"""
        # 结合标题、内容、知识类型和关键概念
        text_parts = [
            knowledge_unit.get("title", ""),
            knowledge_unit.get("content", ""),
            f"知识类型: {knowledge_unit.get('knowledge_type', '')}",
            f"关键概念: {', '.join(knowledge_unit.get('key_concepts', []))}"
        ]
        
        return " ".join([part for part in text_parts if part.strip()])
    
    
    def match_relevant_knowledge(self, match_context: Dict, top_k: int = 5) -> List[Dict]:
        """匹配相关的专家知识"""
        if self.embeddings is None or self.knowledge_units is None:
            return []
        
        # 构建比赛上下文的文本表示
        match_text = self._construct_match_text(match_context)
        
        # 调试：打印生成的上下文
        print(f"\n调试：生成的比赛上下文")
        print(f"  上下文长度: {len(match_text)}")
        print(f"  上下文内容: {match_text[:500]}...")
        
        # 生成比赛上下文的嵌入
        match_embedding = self.generate_embedding(match_text)
        
        # 计算余弦相似度
        similarities = cosine_similarity([match_embedding], self.embeddings)[0]
        
        # 获取最相关的知识单元索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 调试：打印相似度信息
        print(f"调试：相似度信息")
        print(f"  最高相似度: {max(similarities):.4f}")
        print(f"  平均相似度: {np.mean(similarities):.4f}")
        print(f"  相似度前{top_k}: {[f'{similarities[idx]:.4f}' for idx in top_indices]}")
        
        # 构建结果
        relevant_knowledge = []
        for idx in top_indices:
            relevance_score = float(similarities[idx])
            if relevance_score > 0.05:  # 进一步降低相似度阈值，提高匹配成功率
                relevant_knowledge.append({
                    "index": idx,
                    "unit": self.knowledge_units[idx],
                    "relevance_score": relevance_score
                })
        
        return relevant_knowledge
    
    
    def _construct_match_text(self, match_context: Dict) -> str:
        """构建比赛上下文的文本表示"""
        # 使用最新的上下文生成器生成全面的上下文，包含赛果信息用于调试
        return self.context_generator.generate_context(match_context, 'knowledge_matching', include_result=True)
    
    
    def get_enhanced_match_result(self, match_context: Dict, top_k: int = 5) -> Dict:
        """获取增强的匹配结果，包括语义分析"""
        # 获取相关知识
        relevant_knowledge = self.match_relevant_knowledge(match_context, top_k)
        
        # 分析匹配结果
        knowledge_analysis = self._analyze_match_results(relevant_knowledge)
        
        return {
            "relevant_knowledge": relevant_knowledge,
            "match_analysis": knowledge_analysis,
            "total_matches": len(relevant_knowledge)
        }
    
    
    def _analyze_match_results(self, relevant_knowledge: List[Dict]) -> Dict:
        """分析匹配结果"""
        if not relevant_knowledge:
            return {
                "knowledge_type_distribution": {},
                "average_relevance_score": 0.0,
                "highest_relevance_score": 0.0
            }
        
        # 统计知识类型分布
        type_distribution = {}
        total_relevance = 0.0
        highest_relevance = 0.0
        
        for knowledge in relevant_knowledge:
            knowledge_type = knowledge["unit"]["knowledge_type"]
            relevance_score = knowledge["relevance_score"]
            
            # 更新类型分布
            type_distribution[knowledge_type] = type_distribution.get(knowledge_type, 0) + 1
            
            # 更新总相关性
            total_relevance += relevance_score
            
            # 更新最高相关性
            if relevance_score > highest_relevance:
                highest_relevance = relevance_score
        
        return {
            "knowledge_type_distribution": type_distribution,
            "average_relevance_score": total_relevance / len(relevant_knowledge),
            "highest_relevance_score": highest_relevance
        }


def main():
    """测试基于Qwen的知识匹配器"""
    print("=== 基于Qwen的智能知识匹配器测试 ===")
    
    config = {
        "knowledge_matching": {
            "enabled": True,
            "top_k": 5
        }
    }
    
    try:
        # 创建匹配器
        matcher = QwenKnowledgeMatcher(config)
        
        # 测试数据
        test_matches = [
            {
                "home_team": "曼城",
                "away_team": "利物浦",
                "home_win_odds": 1.8,
                "draw_odds": 3.4,
                "away_win_odds": 4.2
            },
            {
                "home_team": "曼联",
                "away_team": "切尔西",
                "home_win_odds": 2.5,
                "draw_odds": 3.2,
                "away_win_odds": 2.8
            }
        ]
        
        for i, match in enumerate(test_matches, 1):
            print(f"\n{'='*60}")
            print(f"测试比赛 {i}: {match['home_team']} vs {match['away_team']}")
            print(f"赔率: 主胜{match['home_win_odds']} 平局{match['draw_odds']} 客胜{match['away_win_odds']}")
            
            # 获取增强的匹配结果
            result = matcher.get_enhanced_match_result(match, top_k=3)
            
            print(f"\n匹配结果:")
            print(f"- 匹配到 {result['total_matches']} 条相关知识")
            print(f"- 平均相关度: {result['match_analysis']['average_relevance_score']:.3f}")
            print(f"- 最高相关度: {result['match_analysis']['highest_relevance_score']:.3f}")
            print(f"- 知识类型分布: {result['match_analysis']['knowledge_type_distribution']}")
            
            # 打印匹配的知识
            for j, knowledge in enumerate(result['relevant_knowledge'], 1):
                print(f"\n  匹配 {j} (相关度: {knowledge['relevance_score']:.3f}):")
                print(f"    标题: {knowledge['unit']['title'][:50]}...")
                print(f"    类型: {knowledge['unit']['knowledge_type']}")
                print(f"    内容: {knowledge['unit']['content'][:100]}...")
        
        print(f"\n{'='*60}")
        print("✅ 基于Qwen的智能知识匹配器测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()