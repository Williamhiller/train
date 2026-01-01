import json
import os
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import PyPDF2
from io import BytesIO
import requests
from transformers import AutoTokenizer
import torch


class ExpertDataProcessor:
    """专家数据处理类，用于处理PDF专家分析"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.max_length = config.get("max_sequence_length", 512)
        
        # 专家分析配置
        self.expert_config = config.get("features", {}).get("expert_analysis", {})
        self.use_llm_for_analysis = self.expert_config.get("use_llm_for_analysis", True)
        self.pdf_directory = self.expert_config.get("pdf_directory", "/Users/Williamhiler/Documents/my-project/train/pdf")
        self.analysis_prompt_template = self.expert_config.get("analysis_prompt_template", 
            "基于以下专家分析，提取足球比赛预测的关键规则和模式：{expert_text}")
        self.max_rules_per_category = self.expert_config.get("max_rules_per_category", 10)
        
    def load_model(self, model_path: str):
        """加载模型和分词器
        
        Args:
            model_path: 模型路径
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 设置设备
        device = self.config.get("qwen", {}).get("device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def split_pdf_into_chapters(self, pdf_content: str) -> List[str]:
        """将PDF内容分割成章节
        
        Args:
            pdf_content: PDF内容
            
        Returns:
            章节列表
        """
        # 尝试识别章节标题
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章',
            r'第[一二三四五六七八九十\d]+节',
            r'\d+\.\s*[^\n]+',
            r'[一二三四五六七八九十]、[^\n]+'
        ]
        
        chapters = []
        current_chapter = ""
        
        lines = pdf_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是章节标题
            is_chapter_title = False
            for pattern in chapter_patterns:
                if re.match(pattern, line):
                    # 保存当前章节
                    if current_chapter:
                        chapters.append(current_chapter)
                    current_chapter = line + "\n"
                    is_chapter_title = True
                    break
            
            if not is_chapter_title:
                current_chapter += line + "\n"
        
        # 添加最后一个章节
        if current_chapter:
            chapters.append(current_chapter)
            
        # 如果没有检测到章节，则按段落分割
        if not chapters:
            paragraphs = pdf_content.split('\n\n')
            # 合并短段落
            chapters = []
            current_chapter = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                if len(current_chapter) + len(para) < 2000:  # 章节最大长度
                    current_chapter += para + "\n\n"
                else:
                    if current_chapter:
                        chapters.append(current_chapter)
                    current_chapter = para + "\n\n"
            
            if current_chapter:
                chapters.append(current_chapter)
        
        return chapters
    
    def extract_expert_rules(self, chapter_text: str) -> Dict[str, List[str]]:
        """从章节文本中提取专家规则
        
        Args:
            chapter_text: 章节文本
            
        Returns:
            专家规则字典
        """
        rules = {
            "odds_rules": [],
            "team_form_rules": [],
            "match_context_rules": [],
            "general_rules": []
        }
        
        # 规则提取模式
        odds_patterns = [
            r'赔率.*?[：:]\s*([^\n。]+)',
            r'欧赔.*?[：:]\s*([^\n。]+)',
            r'亚盘.*?[：:]\s*([^\n。]+)'
        ]
        
        form_patterns = [
            r'状态.*?[：:]\s*([^\n。]+)',
            r'近况.*?[：:]\s*([^\n。]+)',
            r'战绩.*?[：:]\s*([^\n。]+)'
        ]
        
        context_patterns = [
            r'主场.*?[：:]\s*([^\n。]+)',
            r'客场.*?[：:]\s*([^\n。]+)',
            r'对战.*?[：:]\s*([^\n。]+)'
        ]
        
        # 提取赔率规则
        for pattern in odds_patterns:
            matches = re.findall(pattern, chapter_text)
            for match in matches:
                if len(match) > 10:  # 过滤太短的匹配
                    rules["odds_rules"].append(match)
        
        # 提取状态规则
        for pattern in form_patterns:
            matches = re.findall(pattern, chapter_text)
            for match in matches:
                if len(match) > 10:
                    rules["team_form_rules"].append(match)
        
        # 提取上下文规则
        for pattern in context_patterns:
            matches = re.findall(pattern, chapter_text)
            for match in matches:
                if len(match) > 10:
                    rules["match_context_rules"].append(match)
        
        # 提取一般规则
        general_patterns = [
            r'注意.*?[：:]\s*([^\n。]+)',
            r'建议.*?[：:]\s*([^\n。]+)',
            r'技巧.*?[：:]\s*([^\n。]+)'
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, chapter_text)
            for match in matches:
                if len(match) > 10:
                    rules["general_rules"].append(match)
        
        return rules
    
    def process_pdf_files(self, pdf_dir: str = None) -> Dict[str, Dict]:
        """处理PDF文件
        
        Args:
            pdf_dir: PDF目录路径，如果为None则使用配置中的路径
            
        Returns:
            处理后的专家数据
        """
        if pdf_dir is None:
            pdf_dir = self.pdf_directory
        
        expert_data = {}
        
        if not os.path.exists(pdf_dir):
            print(f"PDF directory not found: {pdf_dir}")
            return expert_data
        
        # 遍历PDF文件
        for pdf_file in os.listdir(pdf_dir):
            if not pdf_file.endswith('.pdf'):
                continue
                
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing PDF: {pdf_file}")
            
            # 提取文本
            pdf_content = self.extract_text_from_pdf(pdf_path)
            if not pdf_content:
                continue
            
            # 分割章节
            chapters = self.split_pdf_into_chapters(pdf_content)
            
            # 提取规则
            all_rules = {
                "odds_rules": [],
                "team_form_rules": [],
                "match_context_rules": [],
                "general_rules": []
            }
            
            for chapter in chapters:
                # 使用LLM提取规则
                if self.use_llm_for_analysis and self.model:
                    chapter_rules = self.extract_expert_rules_with_llm(chapter)
                else:
                    chapter_rules = self.extract_expert_rules(chapter)
                    
                for rule_type, rules in chapter_rules.items():
                    all_rules[rule_type].extend(rules)
            
            # 去重
            for rule_type in all_rules:
                all_rules[rule_type] = list(set(all_rules[rule_type]))
                # 限制规则数量
                all_rules[rule_type] = all_rules[rule_type][:self.max_rules_per_category]
            
            # 保存专家数据
            expert_data[pdf_file] = {
                "content": pdf_content,
                "chapters": chapters,
                "rules": all_rules,
                "processed_at": datetime.now().isoformat()
            }
        
        return expert_data
    
    def create_expert_features(self, match_data: Dict, expert_rules: Dict) -> Dict[str, str]:
        """根据比赛数据和专家规则创建专家特征
        
        Args:
            match_data: 比赛数据
            expert_rules: 专家规则
            
        Returns:
            专家特征字典
        """
        # 提取比赛关键信息
        home_team = match_data.get("home_team", "")
        away_team = match_data.get("away_team", "")
        home_odds = match_data.get("home_win_odds", 0)
        draw_odds = match_data.get("draw_odds", 0)
        away_odds = match_data.get("away_win_odds", 0)
        
        # 构建文本特征
        features = {
            "team_recent_form": f"{home_team}近期状态：{match_data.get('home_form_string', '')}，{away_team}近期状态：{match_data.get('away_form_string', '')}",
            "expert_analysis": "",
            "match_context": f"{home_team}主场对阵{away_team}客场，主胜赔率{home_odds}，平局赔率{draw_odds}，客胜赔率{away_odds}"
        }
        
        # 根据专家规则生成分析
        analysis_parts = []
        
        # 赔率分析
        for rule in expert_rules.get("odds_rules", []):
            if self._rule_matches(rule, match_data):
                analysis_parts.append(rule)
        
        # 状态分析
        for rule in expert_rules.get("team_form_rules", []):
            if self._rule_matches(rule, match_data):
                analysis_parts.append(rule)
        
        # 上下文分析
        for rule in expert_rules.get("match_context_rules", []):
            if self._rule_matches(rule, match_data):
                analysis_parts.append(rule)
        
        # 一般分析
        for rule in expert_rules.get("general_rules", [])[:3]:  # 限制一般规则数量
            analysis_parts.append(rule)
        
        features["expert_analysis"] = "。".join(analysis_parts)
        
        return features
    
    def analyze_with_llm(self, text: str) -> str:
        """使用LLM分析专家文本
        
        Args:
            text: 专家文本
            
        Returns:
            LLM分析结果
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        
        # 构建提示
        prompt = self.analysis_prompt_template.format(expert_text=text)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.get("qwen", {}).get("temperature", 0.7),
                top_p=self.config.get("qwen", {}).get("top_p", 0.9),
                do_sample=True
            )
        
        # 解码输出
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取分析结果
        analysis_start = result.find("：") + 1 if "：" in result else 0
        return result[analysis_start:].strip()
    
    def extract_expert_rules_with_llm(self, expert_text: str) -> Dict[str, List[str]]:
        """使用LLM从专家文本中提取规则
        
        Args:
            expert_text: 专家文本
            
        Returns:
            提取的规则字典
        """
        if self.use_llm_for_analysis and self.model:
            # 使用LLM分析
            llm_result = self.analyze_with_llm(expert_text)
            
            # 从LLM结果中提取规则
            return self._parse_llm_result(llm_result)
        else:
            # 使用传统方法
            return self.extract_expert_rules(expert_text)
    
    def _parse_llm_result(self, llm_result: str) -> Dict[str, List[str]]:
        """解析LLM结果
        
        Args:
            llm_result: LLM生成的结果
            
        Returns:
            解析后的规则字典
        """
        rules = {
            "odds_rules": [],
            "team_form_rules": [],
            "match_context_rules": [],
            "general_rules": []
        }
        
        # 简单的规则分类（根据关键词）
        lines = llm_result.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('---'):
                continue
                
            # 分类规则
            if any(keyword in line for keyword in ["赔率", "欧赔", "亚盘", "盘口", "水位"]):
                rules["odds_rules"].append(line)
            elif any(keyword in line for keyword in ["状态", "近况", "战绩", "表现", "连胜", "连败"]):
                rules["team_form_rules"].append(line)
            elif any(keyword in line for keyword in ["主场", "客场", "对阵", "交锋", "历史"]):
                rules["match_context_rules"].append(line)
            else:
                rules["general_rules"].append(line)
        
        # 限制每个类别的规则数量
        for rule_type in rules:
            rules[rule_type] = rules[rule_type][:self.max_rules_per_category]
        
        return rules
    
    def _rule_matches(self, rule: str, match_data: Dict) -> bool:
        """检查规则是否匹配比赛数据
        
        Args:
            rule: 专家规则
            match_data: 比赛数据
            
        Returns:
            是否匹配
        """
        # 简单的匹配逻辑，可以根据需要扩展
        rule_lower = rule.lower()
        match_data_lower = {k.lower(): v for k, v in match_data.items()}
        
        # 检查是否包含相关关键词
        if any(keyword in rule_lower for keyword in ["赔率", "欧赔", "亚盘"]):
            # 赔率规则：检查是否包含赔率相关数据
            return any(key in match_data_lower for key in ["home_win_odds", "draw_odds", "away_win_odds"])
        
        # 状态相关规则
        if any(keyword in rule_lower for keyword in ["状态", "近况", "战绩"]):
            # 状态规则：检查是否包含状态相关数据
            return any(key in match_data_lower for key in ["home_form", "away_form", "home_form_points", "away_form_points"])
        
        # 主客场相关规则
        if any(keyword in rule_lower for keyword in ["主场", "客场"]):
            # 主客场规则：检查是否包含主客场相关数据
            return True
        
        # 其他情况：检查是否包含相关关键词
        for key in match_data_lower:
            if key in rule_lower:
                return True
        
        return False
    
    def tokenize_text(self, text: str) -> Dict:
        """分词文本
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def process_expert_data(self, pdf_dir: str, output_path: str):
        """处理专家数据并保存
        
        Args:
            pdf_dir: PDF目录路径
            output_path: 输出路径
        """
        # 处理PDF文件
        expert_data = self.process_pdf_files(pdf_dir)
        
        # 保存处理结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expert_data, f, ensure_ascii=False, indent=2)
        
        print(f"Expert data saved to {output_path}")
        return expert_data