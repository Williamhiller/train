#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家知识预处理模块
用于将PDF专家文档转化为结构化的知识库
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


class ExpertKnowledgePreprocessor:
    """专家知识预处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.expert_config = config.get("features", {}).get("expert_analysis", {})
        self.pdf_directory = self.expert_config.get("pdf_directory", "/Users/Williamhiler/Documents/my-project/train/pdf")
        self.output_dir = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge"
        
        # 知识分类体系
        self.knowledge_categories = {
            "philosophy": "赔率的哲学思维和认知方法",
            "fundamentals": "赔率分析的基础理论",
            "techniques": "具体的分析技巧和判断标准",
            "patterns": "常见的赔率模式和异常情况",
            "practical": "实战应用和案例分析",
            "psychology": "心理分析和行为模式",
            "risk_management": "风险控制和资金管理"
        }
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_expert_texts(self) -> Dict[str, str]:
        """加载专家文本数据"""
        expert_file = "/Users/Williamhiler/Documents/my-project/train/train-data/expert/pdf_texts.json"
        
        if not os.path.exists(expert_file):
            print(f"专家数据文件不存在: {expert_file}")
            return {}
        
        try:
            with open(expert_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            return {}
    
    
    def extract_knowledge_units(self, text: str, source_doc: str) -> List[Dict]:
        """从文本中提取知识单元"""
        knowledge_units = []
        
        # 按页面分割
        pages = re.split(r'=== 第 \d+ 页 ===', text)
        
        for page_num, page_content in enumerate(pages, 1):
            if not page_content.strip():
                continue
            
            # 提取标题和段落
            sections = self._extract_sections(page_content)
            
            for section in sections:
                unit = {
                    "source_document": source_doc,
                    "page_number": page_num,
                    "content": section["content"],
                    "title": section.get("title", ""),
                    "knowledge_type": self._classify_knowledge_type(section["content"], source_doc),
                    "key_concepts": self._extract_key_concepts(section["content"]),
                    "practical_value": self._assess_practical_value(section["content"]),
                    "context_hints": self._extract_context_hints(section["content"]),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                
                knowledge_units.append(unit)
        
        return knowledge_units
    
    
    def _extract_sections(self, page_content: str) -> List[Dict]:
        """提取页面中的章节段落"""
        sections = []
        
        # 按章节标题分割
        chapter_pattern = r'第[一二三四五六七八九十d]+章\s*[^\n]+|第[一二三四五六七八九十d]+节\s*[^\n]+'
        
        # 如果找到章节标题，按章节分割
        if re.search(chapter_pattern, page_content):
            parts = re.split(chapter_pattern, page_content)
            titles = re.findall(chapter_pattern, page_content)
            
            for i, part in enumerate(parts):
                if part.strip():
                    title = titles[i-1] if i > 0 and i <= len(titles) else ""
                    sections.append({
                        "title": title.strip(),
                        "content": part.strip()
                    })
        else:
            # 按段落分割
            paragraphs = [p.strip() for p in page_content.split('\n') if p.strip()]
            current_section = {"title": "", "content": ""}
            
            for para in paragraphs:
                # 如果是标题行
                if len(para) < 50 and ("。" not in para or "，" not in para):
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = {"title": para, "content": ""}
                else:
                    current_section["content"] += para + "\n"
            
            if current_section["content"]:
                sections.append(current_section)
        
        return sections
    
    
    def _classify_knowledge_type(self, content: str, source_doc: str) -> str:
        """分类知识类型"""
        content_lower = content.lower()
        
        # 哲学思维类关键词
        philosophy_keywords = ["悟", "禅", "道", "哲学", "思维", "认知", "中庸", "阴阳"]
        if any(keyword in content_lower for keyword in philosophy_keywords):
            return "philosophy"
        
        # 基础理论类
        fundamentals_keywords = ["基础", "理论", "原理", "概念", "定义", "本质"]
        if any(keyword in content_lower for keyword in fundamentals_keywords):
            return "fundamentals"
        
        # 实战技巧类
        techniques_keywords = ["技巧", "方法", "步骤", "手法", "招式", "解盘", "分析"]
        if any(keyword in content_lower for keyword in techniques_keywords):
            return "techniques"
        
        # 模式识别类
        patterns_keywords = ["模式", "形态", "特征", "规律", "典型", "常见"]
        if any(keyword in content_lower for keyword in patterns_keywords):
            return "patterns"
        
        # 心理分析类
        psychology_keywords = ["心理", "心态", "情绪", "思维", "认知", "行为"]
        if any(keyword in content_lower for keyword in psychology_keywords):
            return "psychology"
        
        # 风险管理类
        risk_keywords = ["风险", "控制", "管理", "止损", "资金", "仓位"]
        if any(keyword in content_lower for keyword in risk_keywords):
            return "risk_management"
        
        # 默认分类
        return "practical"
    
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """提取关键概念"""
        concepts = []
        
        # 赔率相关概念
        odds_concepts = re.findall(r'欧赔|赔率|盘口|水位|凯利|赔付率|返还率', content)
        concepts.extend(odds_concepts)
        
        # 分析手法概念
        technique_concepts = re.findall(r'拉力|引力|阻力|想象力|迷惑力|平衡点|极限', content)
        concepts.extend(technique_concepts)
        
        # 数值区间概念
        number_concepts = re.findall(r'\d+\.\d+|\d+', content)
        # 只保留看起来像是赔率或重要数值的
        for num in number_concepts:
            try:
                val = float(num)
                if 1.0 <= val <= 10.0:  # 合理的赔率范围
                    concepts.append(f"数值{num}")
            except:
                continue
        
        # 去重并返回前10个最重要的概念
        unique_concepts = list(set(concepts))
        return unique_concepts[:10]
    
    
    def _assess_practical_value(self, content: str) -> Dict[str, float]:
        """评估实用价值"""
        scores = {
            "actionability": 0.0,  # 可操作性
            "specificity": 0.0,    # 具体性
            "uniqueness": 0.0,     # 独特性
            "clarity": 0.0         # 清晰度
        }
        
        # 可操作性评分（是否包含具体的操作指导）
        action_keywords = ["应该", "需要", "必须", "可以", "步骤", "方法", "技巧"]
        action_score = sum(1 for keyword in action_keywords if keyword in content) / len(action_keywords)
        scores["actionability"] = min(action_score * 2, 1.0)  # 最高1.0
        
        # 具体性评分（是否包含具体数值或案例）
        number_count = len(re.findall(r'\d+\.?\d*', content))
        specific_examples = len(re.findall(r'例如|比如|案例|实例', content))
        scores["specificity"] = min((number_count * 0.1 + specific_examples * 0.3), 1.0)
        
        # 独特性评分（是否包含独特的见解或概念）
        unique_terms = ["禅", "悟", "中庸", "阴阳", "极限平衡", "心理跷跷板"]
        uniqueness_score = sum(1 for term in unique_terms if term in content) / len(unique_terms)
        scores["uniqueness"] = uniqueness_score
        
        # 清晰度评分（句子结构清晰度）
        sentences = re.split(r'[。！？]', content)
        clear_sentences = sum(1 for s in sentences if len(s.strip()) > 10 and len(s.strip()) < 100)
        total_sentences = max(len(sentences), 1)
        scores["clarity"] = clear_sentences / total_sentences
        
        return scores
    
    
    def _extract_context_hints(self, content: str) -> Dict[str, List[str]]:
        """提取上下文提示"""
        hints = {
            "applicable_scenarios": [],  # 适用场景
            "prerequisites": [],        # 前提条件
            "warnings": [],            # 注意事项
            "related_concepts": []     # 相关概念
        }
        
        # 提取适用场景
        scenario_patterns = [
            r"当(.*?)时",
            r"在(.*?)情况下",
            r"如果(.*?)的话"
        ]
        
        for pattern in scenario_patterns:
            matches = re.findall(pattern, content)
            hints["applicable_scenarios"].extend(matches)
        
        # 提取前提条件
        prerequisite_patterns = [
            r"需要(.*?)才能",
            r"必须(.*?)才",
            r"前提是(.*?)"
        ]
        
        for pattern in prerequisite_patterns:
            matches = re.findall(pattern, content)
            hints["prerequisites"].extend(matches)
        
        # 提取注意事项
        warning_keywords = ["注意", "小心", "谨慎", "风险", "切忌", "不要"]
        for keyword in warning_keywords:
            if keyword in content:
                # 提取包含关键词的句子
                sentences = re.findall(r'[^。！？]*' + keyword + r'[^。！？]*[。！？]', content)
                hints["warnings"].extend(sentences)
        
        # 提取相关概念
        related_patterns = [
            r"与(.*?)相关",
            r"涉及到(.*?)",
            r"包括(.*?)等"
        ]
        
        for pattern in related_patterns:
            matches = re.findall(pattern, content)
            hints["related_concepts"].extend(matches)
        
        return hints
    
    
    def process_all_documents(self) -> Dict[str, any]:
        """处理所有专家文档"""
        print("=== 开始专家知识预处理 ===")
        
        # 加载专家文本
        expert_texts = self.load_expert_texts()
        if not expert_texts:
            print("没有可用的专家数据")
            return {}
        
        print(f"加载了 {len(expert_texts)} 个专家文档")
        
        all_knowledge_units = []
        document_stats = {}
        
        # 处理每个文档
        for doc_name, doc_content in expert_texts.items():
            print(f"正在处理文档: {doc_name}")
            
            # 提取知识单元
            knowledge_units = self.extract_knowledge_units(doc_content, doc_name)
            
            # 统计信息
            doc_stats = {
                "total_pages": len(re.findall(r'=== 第 \d+ 页 ===', doc_content)),
                "knowledge_units": len(knowledge_units),
                "categories": {}
            }
            
            # 按类型统计
            for unit in knowledge_units:
                category = unit["knowledge_type"]
                doc_stats["categories"][category] = doc_stats["categories"].get(category, 0) + 1
            
            document_stats[doc_name] = doc_stats
            all_knowledge_units.extend(knowledge_units)
            
            print(f"  - 提取了 {len(knowledge_units)} 个知识单元")
            print(f"  - 知识类型分布: {doc_stats['categories']}")
        
        # 构建知识库
        knowledge_base = {
            "metadata": {
                "total_documents": len(expert_texts),
                "total_knowledge_units": len(all_knowledge_units),
                "processing_timestamp": datetime.now().isoformat(),
                "document_statistics": document_stats
            },
            "knowledge_units": all_knowledge_units,
            "category_index": self._build_category_index(all_knowledge_units),
            "concept_index": self._build_concept_index(all_knowledge_units)
        }
        
        # 保存知识库
        self.save_knowledge_base(knowledge_base)
        
        return knowledge_base
    
    
    def _build_category_index(self, knowledge_units: List[Dict]) -> Dict[str, List[int]]:
        """构建分类索引"""
        category_index = {}
        
        for i, unit in enumerate(knowledge_units):
            category = unit["knowledge_type"]
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(i)
        
        return category_index
    
    
    def _build_concept_index(self, knowledge_units: List[Dict]) -> Dict[str, List[int]]:
        """构建概念索引"""
        concept_index = {}
        
        for i, unit in enumerate(knowledge_units):
            concepts = unit.get("key_concepts", [])
            for concept in concepts:
                if concept not in concept_index:
                    concept_index[concept] = []
                concept_index[concept].append(i)
        
        return concept_index
    
    
    def save_knowledge_base(self, knowledge_base: Dict):
        """保存知识库"""
        # 主知识库文件
        main_file = os.path.join(self.output_dir, "expert_knowledge_base.json")
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        # 分类索引文件
        category_file = os.path.join(self.output_dir, "category_index.json")
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base["category_index"], f, ensure_ascii=False, indent=2)
        
        # 概念索引文件
        concept_file = os.path.join(self.output_dir, "concept_index.json")
        with open(concept_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base["concept_index"], f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        self.generate_statistics_report(knowledge_base)
        
        print(f"知识库已保存到: {self.output_dir}")
        print(f"主知识库: {len(knowledge_base['knowledge_units'])} 个知识单元")
        print(f"分类索引: {len(knowledge_base['category_index'])} 个分类")
        print(f"概念索引: {len(knowledge_base['concept_index'])} 个概念")
    
    
    def generate_statistics_report(self, knowledge_base: Dict):
        """生成统计报告"""
        report = {
            "processing_summary": {
                "total_knowledge_units": len(knowledge_base["knowledge_units"]),
                "total_categories": len(knowledge_base["category_index"]),
                "total_concepts": len(knowledge_base["concept_index"]),
                "documents_processed": knowledge_base["metadata"]["total_documents"]
            },
            "category_breakdown": {},
            "quality_metrics": {
                "average_actionability": 0.0,
                "average_specificity": 0.0,
                "average_uniqueness": 0.0,
                "average_clarity": 0.0
            },
            "top_concepts": [],
            "document_contribution": {}
        }
        
        # 分类统计
        for category, indices in knowledge_base["category_index"].items():
            category_name = self.knowledge_categories.get(category, category)
            report["category_breakdown"][category_name] = len(indices)
        
        # 概念统计（按出现频率排序）
        concept_counts = [(concept, len(indices)) for concept, indices in knowledge_base["concept_index"].items()]
        concept_counts.sort(key=lambda x: x[1], reverse=True)
        report["top_concepts"] = concept_counts[:20]
        
        # 文档贡献统计
        for doc_name, stats in knowledge_base["metadata"]["document_statistics"].items():
            report["document_contribution"][doc_name] = {
                "knowledge_units": stats["knowledge_units"],
                "categories": len(stats["categories"]),
                "pages_processed": stats["total_pages"]
            }
        
        # 质量指标统计
        total_actionability = 0
        total_specificity = 0
        total_uniqueness = 0
        total_clarity = 0
        
        for unit in knowledge_base["knowledge_units"]:
            practical_value = unit.get("practical_value", {})
            total_actionability += practical_value.get("actionability", 0)
            total_specificity += practical_value.get("specificity", 0)
            total_uniqueness += practical_value.get("uniqueness", 0)
            total_clarity += practical_value.get("clarity", 0)
        
        total_units = len(knowledge_base["knowledge_units"])
        if total_units > 0:
            report["quality_metrics"]["average_actionability"] = total_actionability / total_units
            report["quality_metrics"]["average_specificity"] = total_specificity / total_units
            report["quality_metrics"]["average_uniqueness"] = total_uniqueness / total_units
            report["quality_metrics"]["average_clarity"] = total_clarity / total_units
        
        # 保存报告
        report_file = os.path.join(self.output_dir, "preprocessing_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print("\n=== 知识预处理统计报告 ===")
        print(f"总知识单元数: {report['processing_summary']['total_knowledge_units']}")
        print(f"知识分类数: {report['processing_summary']['total_categories']}")
        print(f"关键概念数: {report['processing_summary']['total_concepts']}")
        print("\n分类分布:")
        for category, count in report["category_breakdown"].items():
            print(f"  {category}: {count}")
        print(f"\n质量指标:")
        for metric, value in report["quality_metrics"].items():
            print(f"  {metric}: {value:.3f}")
        
        return report


def main():
    """主函数"""
    print("=== 专家知识预处理系统 ===")
    
    # 配置
    config = {
        "features": {
            "expert_analysis": {
                "pdf_directory": "/Users/Williamhiler/Documents/my-project/train/pdf"
            }
        }
    }
    
    # 创建预处理器
    preprocessor = ExpertKnowledgePreprocessor(config)
    
    # 处理所有文档
    knowledge_base = preprocessor.process_all_documents()
    
    if knowledge_base:
        print("\n✅ 专家知识预处理完成！")
        print("您现在可以：")
        print("1. 查看生成的知识库文件")
        print("2. 手动调整知识分类和标注")
        print("3. 基于知识库构建智能推理系统")
    else:
        print("\n❌ 专家知识预处理失败！")


if __name__ == "__main__":
    main()