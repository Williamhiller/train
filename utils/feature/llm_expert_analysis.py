#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM驱动的智能专家分析模块
基于大语言模型的专家文字分析，提取高质量的结构化特征
"""

import json
import os
import yaml
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 尝试导入torch和transformers，如果不可用则使用备选方案
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@dataclass
class ExpertAnalysisResult:
    """专家分析结果数据结构"""
    confidence_score: float  # 专家信心评分 (0-1)
    prediction: str  # 预测结果 ('主胜', '平局', '客胜')
    reasoning_quality: float  # 分析逻辑质量 (0-1)
    key_factors: List[str]  # 关键因素列表
    risk_assessment: float  # 风险评估 (0-1)
    sentiment: str  # 情感倾向 ('乐观', '谨慎', '悲观')
    timestamp: str  # 分析时间

class LLMExpertAnalyzer:
    """LLM驱动的专家分析器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化LLM专家分析器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # 设置设备
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        print(f"使用设备: {self.device}")
        print(f"torch可用: {TORCH_AVAILABLE}")
        print(f"transformers可用: {TRANSFORMERS_AVAILABLE}")
        self._initialize_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('expert_analysis_llm', {})
    
    def _initialize_model(self):
        """初始化LLM模型"""
        if not self.config.get('enabled', False):
            print("⚠ LLM专家分析功能未在配置中启用，将使用传统分析方法")
            self.pipeline = None
            return
        
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            print("⚠ torch或transformers不可用，将使用传统分析方法")
            self.pipeline = None
            return
        
        model_name = self.config.get('model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
        cache_dir = self.config.get('cache_dir', './models/cache')
        
        print(f"正在加载模型: {model_name}")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 查找本地模型快照路径
        model_cache_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        local_model_path = None
        
        # 尝试多个可能的本地路径
        possible_paths = [
            model_cache_path,
            os.path.join(cache_dir, "models--Qwen--Qwen2.5-1.5B-Instruct"),  # 回退到1.5B
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # 查找snapshot中的实际路径
                for root, dirs, files in os.walk(path):
                    # 检查关键文件是否存在
                    has_config = 'config.json' in files
                    has_model = any(f in files for f in ['model.safetensors', 'pytorch_model.bin', 'model-00001-of-00002.safetensors'])
                    if has_config and has_model:
                        local_model_path = root
                        break
                if local_model_path:
                    break
        
        try:
            if local_model_path and os.path.exists(local_model_path):
                print(f"✓ 使用本地模型路径: {local_model_path}")
                
                # 内存优化加载 - 针对CPU优化
                print("使用CPU优化配置加载...")
                
                # 加载tokenizer（消耗内存少，先加载）
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=False  # 减少内存使用
                )
                
                # 只加载模型权重，不加载到GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    torch_dtype=torch.float32,  # CPU最稳定的格式
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,  # 关键优化：减少CPU内存占用
                    device_map=None,  # 不自动分配设备
                    offload_folder=None,  # 不使用offload
                    offload_state_dict=False  # 不offload状态字典
                )
                
                # 将模型移动到CPU
                self.model.to('cpu')
                
                # 禁用梯度计算，减少内存使用
                for param in self.model.parameters():
                    param.requires_grad = False
                
                print("✓ 模型权重加载完成，内存优化完成")
            else:
                print("⚠ 本地模型缓存不完整，将使用传统关键词分析方法")
                self.pipeline = None
                return
            
            # 创建推理管道
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,  # 限制生成长度
                max_length=self.config.get('max_length', 256),
                temperature=self.config.get('temperature', 0.3),
                top_p=self.config.get('top_p', 0.8),
                do_sample=self.config.get('do_sample', True)
            )
            
            # 统计模型参数
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"✓ 模型加载成功: {model_name} ({num_params/1e6:.1f}M 参数)")
            
        except Exception as e:
            print(f"⚠ 模型加载失败: {e}")
            print("将使用传统关键词分析方法")
            self.pipeline = None
    
    def analyze_expert_text(self, expert_text: str, match_context: Dict = None, max_retries: int = 3) -> ExpertAnalysisResult:
        """
        使用LLM分析专家文字
        
        参数:
            expert_text: 专家分析文字
            match_context: 比赛上下文信息
            max_retries: 最大重试次数
            
        返回:
            ExpertAnalysisResult: 分析结果
        """
        # 如果没有加载模型，直接使用传统方法
        if not self.pipeline:
            print("使用传统关键词分析方法...")
            return self._fallback_analysis(expert_text)
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(expert_text, match_context)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # 生成分析
                response = self.pipeline(prompt, max_new_tokens=800, do_sample=True)
                
                # 获取生成的文本，排除原始提示
                full_text = response[0]['generated_text']
                generated_text = full_text[len(prompt):].strip()
                
                # 解析结果
                result = self._parse_llm_response(generated_text, expert_text)
                
                # 验证结果有效性
                if self._validate_analysis_result(result):
                    print(f"✓ LLM分析完成，信心评分: {result.confidence_score:.2f}")
                    return result
                else:
                    print(f"⚠ 第{attempt + 1}次尝试: 分析结果验证失败")
                    last_error = "结果验证失败"
                    
            except Exception as e:
                last_error = str(e)
                print(f"⚠ 第{attempt + 1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待后重试
        
        # 所有重试都失败，使用fallback
        print(f"✗ LLM分析失败，使用fallback方法: {last_error}")
        return self._fallback_analysis(expert_text)
    
    def _validate_analysis_result(self, result: ExpertAnalysisResult) -> bool:
        """验证分析结果的有效性"""
        if not (0 <= result.confidence_score <= 1):
            return False
        if not (0 <= result.reasoning_quality <= 1):
            return False
        if not (0 <= result.risk_assessment <= 1):
            return False
        if result.prediction not in ['主胜', '平局', '客胜', '未知']:
            return False
        if result.sentiment not in ['乐观', '谨慎', '悲观']:
            return False
        if not isinstance(result.key_factors, list):
            return False
        return True
    
    def _build_analysis_prompt(self, expert_text: str, match_context: Dict = None) -> str:
        """构建分析提示词"""
        context_info = ""
        if match_context:
            context_info = f"""
比赛信息:
- 主队: {match_context.get('home_team', '未知')}
- 客队: {match_context.get('away_team', '未知')}
- 联赛: {match_context.get('league', '未知')}
- 主胜赔率: {match_context.get('home_odds', 'N/A')}
- 平局赔率: {match_context.get('draw_odds', 'N/A')}
- 客胜赔率: {match_context.get('away_odds', 'N/A')}
"""
        
        prompt = f"""{context_info}
你是一位专业的足球赔率分析专家。请分析以下专家文字，提取结构化的分析特征。

专家分析文字：
"{expert_text}"

请严格按照以下JSON格式输出分析结果（只输出JSON，不要有任何其他文字）：

{{
    "confidence_score": 0.85,
    "prediction": "主胜",
    "reasoning_quality": 0.8,
    "key_factors": ["球队近期状态", "主客场表现", "赔率分析", "历史交锋"],
    "risk_assessment": 0.3,
    "sentiment": "谨慎"
}}

字段说明：
1. confidence_score: 信心评分(0.0-1.0)，1.0表示非常有把握
2. prediction: 预测结果，只能是"主胜"、"平局"或"客胜"中的一个
3. reasoning_quality: 分析逻辑质量(0.0-1.0)
4. key_factors: 关键因素列表，从以下选项中选择2-5个最相关的：
   - "球队近期状态"：球队最近几场的表现
   - "主客场表现"：主场或客场作战能力
   - "赔率分析"：欧赔或亚盘赔率暗示的信息
   - "历史交锋"：两队历史对战记录
   - "伤病情况"：关键球员是否受伤
   - "战意分析"：球队是否有强烈的求胜欲望
   - "联赛排名"：球队在联赛中的排名位置
   - "球队实力对比"：两队整体实力差距
   - "盘口变化"：赔率或盘口的异常变化
   - "基本面分析"：球队的基本情况分析
5. risk_assessment: 风险评估(0.0-1.0)，1.0表示风险很高
6. sentiment: 情感倾向，只能是"乐观"、"谨慎"或"悲观"中的一个

请直接输出JSON对象，确保格式完全正确。
"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, original_text: str) -> ExpertAnalysisResult:
        """解析LLM响应"""
        try:
            # 清理响应文本
            response_text = response_text.strip()
            
            # 尝试多种方式提取JSON
            json_str = None
            
            # 方法1: 查找完整的JSON对象
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > 0:
                json_str = response_text[json_start:json_end]
            else:
                # 方法2: 尝试用正则表达式匹配
                json_match = re.search(r'\{[^{}]*\}', response_text)
                if json_match:
                    json_str = json_match.group()
            
            if not json_str:
                raise ValueError("未找到JSON格式")
            
            # 清理JSON字符串
            json_str = json_str.strip()
            
            # 修复常见的JSON格式问题
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('    ', ' ')
            json_str = re.sub(r'\s+', ' ', json_str)
            
            # 转换为结构化结果
            result_dict = json.loads(json_str)
            
            # 规范化关键因素列表
            key_factors = result_dict.get('key_factors', [])
            if isinstance(key_factors, str):
                key_factors = [key_factors]
            if not isinstance(key_factors, list):
                key_factors = []
            
            return ExpertAnalysisResult(
                confidence_score=float(result_dict.get('confidence_score', 0.5)),
                prediction=result_dict.get('prediction', '未知'),
                reasoning_quality=float(result_dict.get('reasoning_quality', 0.5)),
                key_factors=key_factors,
                risk_assessment=float(result_dict.get('risk_assessment', 0.5)),
                sentiment=result_dict.get('sentiment', '谨慎'),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"尝试解析的文本: {response_text[:200]}...")
            return self._fallback_analysis(original_text)
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return self._fallback_analysis(original_text)
    
    def _fallback_analysis(self, expert_text: str) -> ExpertAnalysisResult:
        """回退到传统分析方法"""
        # 简单的关键词分析作为备选方案
        confidence = 0.5
        key_factors = []
        
        # 关键词匹配 - 信心评分
        if '极高' in expert_text or '极大' in expert_text or '强烈' in expert_text:
            confidence = 0.9
        elif '较高' in expert_text or '较大' in expert_text or '看好' in expert_text:
            confidence = 0.7
        elif '较低' in expert_text or '较小' in expert_text or '谨慎' in expert_text:
            confidence = 0.3
        elif '不确定' in expert_text or '难说' in expert_text:
            confidence = 0.2
        
        # 预测结果提取
        prediction = '未知'
        if '主胜' in expert_text or '主队胜' in expert_text or '主队赢' in expert_text:
            prediction = '主胜'
        elif '平局' in expert_text or '打平' in expert_text or '和局' in expert_text:
            prediction = '平局'
        elif '客胜' in expert_text or '客队胜' in expert_text or '主负' in expert_text:
            prediction = '客胜'
        
        # 提取关键因素
        factor_keywords = {
            '球队近期状态': ['近期', '最近', '状态', '连胜', '连败', '不胜'],
            '主客场表现': ['主场', '客场', '主场优势', '客场作战'],
            '赔率分析': ['赔率', '水位', '盘口', '欧赔', '亚盘'],
            '历史交锋': ['交锋', '历史', '往绩', '对战'],
            '伤病情况': ['伤病', '缺阵', '伤停', '缺席'],
            '战意分析': ['战意', '求胜', '战意强烈', '无欲无求'],
            '联赛排名': ['排名', '积分', '榜尾', '榜首'],
            '球队实力对比': ['实力', '实力差距', '强队', '弱队'],
            '盘口变化': ['变盘', '水位变化', '升盘', '降盘'],
            '基本面分析': ['基本面', '球队', '阵容', '战术']
        }
        
        for factor, keywords in factor_keywords.items():
            for keyword in keywords:
                if keyword in expert_text:
                    key_factors.append(factor)
                    break
        
        # 如果没有提取到关键因素，添加默认因素
        if not key_factors:
            key_factors = ['基本面分析']
        
        # 去重并限制数量
        key_factors = list(dict.fromkeys(key_factors))[:5]
        
        # 情感分析
        sentiment = '谨慎'
        if '乐观' in expert_text or '看好' in expert_text or '信心十足' in expert_text:
            sentiment = '乐观'
        elif '悲观' in expert_text or '不看好' in expert_text or '风险' in expert_text:
            sentiment = '悲观'
        
        # 风险评估
        risk_assessment = 0.5
        if '高风险' in expert_text or '风险较大' in expert_text:
            risk_assessment = 0.8
        elif '低风险' in expert_text or '风险较小' in expert_text or '稳健' in expert_text:
            risk_assessment = 0.2
        
        return ExpertAnalysisResult(
            confidence_score=confidence,
            prediction=prediction,
            reasoning_quality=0.5,
            key_factors=key_factors,
            risk_assessment=risk_assessment,
            sentiment=sentiment,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def batch_analyze(self, expert_texts: List[str], match_contexts: List[Dict] = None) -> List[ExpertAnalysisResult]:
        """
        批量分析专家文字
        
        参数:
            expert_texts: 专家文字列表
            match_contexts: 比赛上下文列表
            
        返回:
            List[ExpertAnalysisResult]: 分析结果列表
        """
        results = []
        
        for i, expert_text in enumerate(expert_texts):
            context = match_contexts[i] if match_contexts and i < len(match_contexts) else None
            result = self.analyze_expert_text(expert_text, context)
            results.append(result)
        
        return results
    
    def get_analysis_summary(self, results: List[ExpertAnalysisResult]) -> Dict:
        """获取分析结果摘要"""
        if not results:
            return {}
        
        total_confidence = sum(r.confidence_score for r in results)
        avg_confidence = total_confidence / len(results)
        
        prediction_counts = {}
        for result in results:
            prediction = result.prediction
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        
        sentiment_counts = {}
        for result in results:
            sentiment = result.sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            'total_analyses': len(results),
            'average_confidence': avg_confidence,
            'prediction_distribution': prediction_counts,
            'sentiment_distribution': sentiment_counts,
            'high_confidence_count': sum(1 for r in results if r.confidence_score > 0.7)
        }

def test_llm_expert_analysis():
    """测试LLM专家分析功能"""
    print("=== 测试LLM专家分析功能 ===")
    
    # 初始化分析器
    analyzer = LLMExpertAnalyzer()
    
    # 测试案例
    test_cases = [
        {
            'text': "主队近期状态出色，主场优势明显，看好主胜",
            'context': {
                'home_team': '曼联',
                'away_team': '利物浦',
                'home_odds': 2.5,
                'draw_odds': 3.2,
                'away_odds': 2.8
            }
        },
        {
            'text': "客队实力占优，但赔率过于保守，建议谨慎投注客胜",
            'context': {
                'home_team': '阿森纳',
                'away_team': '切尔西',
                'home_odds': 3.1,
                'draw_odds': 3.3,
                'away_odds': 2.2
            }
        }
    ]
    
    print(f"配置信息: {analyzer.config}")
    print(f"设备: {analyzer.device}")
    print(f"模型状态: {'已加载' if analyzer.pipeline else '未加载'}")
    
    # 执行分析
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试案例 {i+1} ---")
        print(f"专家文字: {test_case['text']}")
        
        result = analyzer.analyze_expert_text(
            test_case['text'], 
            test_case['context']
        )
        
        print(f"预测结果: {result.prediction}")
        print(f"信心评分: {result.confidence_score:.2f}")
        print(f"逻辑质量: {result.reasoning_quality:.2f}")
        print(f"情感倾向: {result.sentiment}")
        print(f"风险评估: {result.risk_assessment:.2f}")
        print(f"关键因素: {result.key_factors}")

if __name__ == "__main__":
    test_llm_expert_analysis()