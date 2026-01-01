import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
import os


class MatchDataset(Dataset):
    """比赛数据集类"""
    
    def __init__(self, 
                 structured_data: pd.DataFrame, 
                 text_data: List[Dict], 
                 feature_columns: List[str],
                 label_column: str = "result"):
        """
        初始化数据集
        
        Args:
            structured_data: 结构化数据DataFrame
            text_data: 文本数据列表
            feature_columns: 特征列名
            label_column: 标签列名
        """
        self.structured_data = structured_data
        self.text_data = text_data
        self.feature_columns = feature_columns
        self.label_column = label_column
        
        # 检查数据一致性
        assert len(structured_data) == len(text_data), "Structured data and text data must have the same length"
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.structured_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含结构化数据、文本和标签的字典
        """
        # 获取结构化数据
        structured_features = self.structured_data.iloc[idx][self.feature_columns].values.astype(np.float32)
        label = self.structured_data.iloc[idx][self.label_column]
        
        # 获取文本数据
        text_data = self.text_data[idx]
        
        return {
            "structured_features": torch.tensor(structured_features, dtype=torch.float),
            "text_features": text_data,
            "label": torch.tensor(label, dtype=torch.long)
        }


class V5DataLoader:
    """V5模型数据加载器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config["data"]["batch_size"]
        self.train_val_split = config["data"]["train_val_split"]
        
    def prepare_data(self, data_path: str, expert_data_path: str = None, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据，支持批次处理
        
        Args:
            data_path: 原始数据路径
            expert_data_path: 专家数据路径
            batch_size: 批次大小，为None时处理所有数据
            
        Returns:
            训练和验证数据加载器
        """
        # 导入数据处理器
        from .match_data_processor import MatchDataProcessor
        from .context_generator import ContextGenerator
        from utils.expert_knowledge.intelligent_rule_matcher import IntelligentRuleMatcher
        
        # 获取历史数据配置
        historical_config = self.config.get("data", {}).get("historical_data", {})
        use_historical = historical_config.get("use_historical", True)
        seasons = historical_config.get("seasons", ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"])
        
        # 处理比赛数据，支持批次处理
        match_processor = MatchDataProcessor(self.config)
        match_data = match_processor.process_data(data_path, batch_size=batch_size)
        
        # 如果返回的是字典（按赛季分批），合并所有赛季数据
        if isinstance(match_data, dict):
            print(f"Merging data from {len(match_data)} seasons...")
            all_data = []
            for season, season_df in match_data.items():
                all_data.append(season_df)
            match_data = pd.concat(all_data, ignore_index=True)
            print(f"Total matches after merging: {len(match_data)}")
        
        # 从预生成的JSON文件中加载上下文
        print("Loading pre-generated contexts...")
        
        # 构建上下文文件路径
        context_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "contexts")
        
        # 尝试加载按赛季分批的上下文文件
        context_file_path = os.path.join(context_dir, "contexts_knowledge_matching_knowledge_matching_without_result.json")
        
        # 检查上下文文件是否存在
        if not os.path.exists(context_file_path):
            # 如果不存在，尝试加载旧的合并文件
            context_file_path = os.path.join(context_dir, "contexts_knowledge_matching_result_False.json")
            if not os.path.exists(context_file_path):
                raise FileNotFoundError(f"上下文文件不存在: {context_file_path}")
            print(f"使用合并的上下文文件: {context_file_path}")
        else:
            print(f"使用按赛季分批的上下文文件: {context_file_path}")
        
        # 加载上下文
        with open(context_file_path, 'r', encoding='utf-8') as f:
            pre_generated_contexts = json.load(f)
        
        print(f"Loaded {len(pre_generated_contexts)} pre-generated contexts")
        
        # 初始化智能规则匹配器
        print("Initializing Intelligent Rule Matcher...")
        # 使用更可靠的路径构建方式
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rules_path = os.path.join(project_root, "data", "expert_knowledge", "expert_rules.json")

        # 如果规则文件不存在，先创建空文件
        if not os.path.exists(rules_path):
            print(f"⚠️  规则文件不存在: {rules_path}")
            print("   将使用基本上下文，不注入专家知识")
            use_expert_knowledge = False
        else:
            try:
                rule_matcher = IntelligentRuleMatcher(self.config, rules_path)
                use_expert_knowledge = True
                print("✅ 智能规则匹配器初始化成功")
            except Exception as e:
                print(f"❌ 初始化规则匹配器失败: {e}")
                print("   将使用基本上下文，不注入专家知识")
                use_expert_knowledge = False
        
        # 创建文本特征 - 直接使用预生成的上下文
        text_data = []
        # 统计找到和未找到的上下文数量
        found_count = 0
        not_found_count = 0
        
        for idx, (_, match) in enumerate(match_data.iterrows()):
            match_dict = match.to_dict()
            
            # 从预生成的上下文中获取匹配的上下文
            # 检查所有可能的比赛ID字段名
            match_id = None
            # 尝试所有可能的比赛ID字段名
            possible_id_keys = ['match_id', 'matchId', 'MatchID', 'id', 'matchid', 'matchID', 'MatchId']
            for key in possible_id_keys:
                if key in match_dict:
                    match_id = str(match_dict[key])
                    break
            else:
                # 如果没有找到，尝试其他可能的字段名
                for key in match_dict:
                    if isinstance(match_dict[key], (int, str)) and len(str(match_dict[key])) > 5:
                        match_id = str(match_dict[key])
                        break
                else:
                    print(f"No match key found for keys: {list(match_dict.keys())[:10]}")
                    match_id = ""
            
            # 从预生成的上下文中获取匹配的上下文
            if match_id and match_id in pre_generated_contexts:
                match_context = pre_generated_contexts[match_id]
                found_count += 1
            else:
                # 如果没有找到，生成一个简单的上下文
                if idx < 3:  # 只打印前3个
                    print(f"Context not found for match_id: {match_id}, available keys: {list(match_dict.keys())[:10]}")
                home_team_name = match_dict.get('homeTeamName', match_dict.get('home_team_name', match_dict.get('homeTeam', '主队')))
                away_team_name = match_dict.get('awayTeamName', match_dict.get('away_team_name', match_dict.get('awayTeam', '客队')))
                match_context = f"比赛ID: {match_id}; 比赛球队: {home_team_name} vs {away_team_name}"
                not_found_count += 1
            
            # 注入专家知识
            if use_expert_knowledge:
                # 匹配相关的专家规则
                matched_rules = rule_matcher.match_rules_for_match(match_dict, top_k=2)
                
                # 格式化专家规则
                expert_knowledge_text = rule_matcher.format_matched_rules(matched_rules)
                
                # 构建包含专家知识的文本特征
                combined_text = f"{match_context}\n\n{expert_knowledge_text}"
            else:
                # 不使用专家知识
                combined_text = match_context
            
            text_data.append(combined_text)
        
        # 打印上下文匹配结果总结
        print(f"上下文匹配总结：找到 {found_count} 个，未找到 {not_found_count} 个")
        
        # 获取特征列
        feature_columns = match_processor.get_feature_columns()
        
        # 分割训练和验证集
        n_samples = len(match_data)
        n_train = int(n_samples * self.train_val_split)
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # 创建训练和验证数据集
        train_structured = match_data.iloc[train_indices].reset_index(drop=True)
        val_structured = match_data.iloc[val_indices].reset_index(drop=True)
        
        train_text = [text_data[i] for i in train_indices]
        val_text = [text_data[i] for i in val_indices]
        
        train_dataset = MatchDataset(train_structured, train_text, feature_columns)
        val_dataset = MatchDataset(val_structured, val_text, feature_columns)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def prepare_test_data(self, data_path: str, expert_data_path: str) -> DataLoader:
        """准备测试数据
        
        Args:
            data_path: 测试数据路径
            expert_data_path: 专家数据路径
            
        Returns:
            测试数据加载器
        """
        # 导入数据处理器
        from .match_data_processor import MatchDataProcessor
        from utils.expert_knowledge.intelligent_expert_reasoner import IntelligentExpertReasoner
        from context_generator import ContextGenerator
        
        # 处理比赛数据
        match_processor = MatchDataProcessor(self.config)
        match_data = match_processor.process_data(data_path)
        
        # 从预生成的JSON文件中加载上下文
        print("Loading pre-generated contexts for test data...")
        
        # 构建上下文文件路径
        context_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "contexts")
        context_file_path = os.path.join(context_dir, "contexts_knowledge_matching_result_False.json")
        
        # 检查上下文文件是否存在
        if not os.path.exists(context_file_path):
            raise FileNotFoundError(f"上下文文件不存在: {context_file_path}")
        
        # 加载上下文
        with open(context_file_path, 'r', encoding='utf-8') as f:
            pre_generated_contexts = json.load(f)
        
        print(f"Loaded {len(pre_generated_contexts)} pre-generated contexts for test data")
        
        # 初始化智能专家推理器（集成了Qwen知识匹配系统）
        print("Initializing Intelligent Expert Reasoner for test data...")
        reasoner = IntelligentExpertReasoner(self.config)
        
        # 创建文本特征 - 从预生成的上下文加载，并添加专家分析
        text_data = []
        # 统计找到和未找到的上下文数量
        found_count = 0
        not_found_count = 0
        
        for _, match in match_data.iterrows():
            match_dict = match.to_dict()
            
            # 从预生成的上下文中获取匹配的上下文
            # 检查所有可能的比赛ID字段名
            match_id = None
            # 尝试所有可能的比赛ID字段名
            possible_id_keys = ['match_id', 'matchId', 'MatchID', 'id', 'matchid', 'matchID', 'MatchId']
            for key in possible_id_keys:
                if key in match_dict:
                    match_id = str(match_dict[key])
                    break
            else:
                # 如果没有找到，尝试其他可能的字段名
                for key in match_dict:
                    if isinstance(match_dict[key], (int, str)) and len(str(match_dict[key])) > 5:
                        match_id = str(match_dict[key])
                        break
                else:
                    print(f"No match key found for keys: {list(match_dict.keys())[:10]}")
                    match_id = ""
            
            # 从预生成的上下文中获取匹配的上下文
            if match_id and match_id in pre_generated_contexts:
                match_context = pre_generated_contexts[match_id]
                found_count += 1
            else:
                # 如果没有找到，生成一个简单的上下文
                print(f"Context not found for match_id: {match_id}, generating simple context...")
                match_context = f"比赛ID: {match_id}; 比赛球队: {match_dict.get('homeTeamName', match_dict.get('home_team_name', match_dict.get('homeTeam', '主队')))} vs {match_dict.get('awayTeamName', match_dict.get('away_team_name', match_dict.get('awayTeam', '客队')))}"
                not_found_count += 1
            
            # 添加专家分析和知识匹配结果
            expert_analysis = reasoner.analyze_match_with_expert_knowledge(match_dict)
            knowledge_summary = reasoner.get_match_knowledge_summary(match_dict)
            
            # 构建专家分析文本
            expert_analysis_text = f"专家分析：基于{expert_analysis['expert_knowledge_used']}条相关知识，置信度{expert_analysis['confidence']:.2f}。"
            expert_analysis_text += f"主胜概率{expert_analysis['home_win_prob']:.2f}，平局概率{expert_analysis['draw_prob']:.2f}，客胜概率{expert_analysis['away_win_prob']:.2f}。"
            
            # 构建知识匹配文本
            top_knowledge_text = "关键知识："
            for i, knowledge in enumerate(knowledge_summary['top_knowledge'][:3]):
                top_knowledge_text += f"{i+1}.{knowledge['title']}（{knowledge['knowledge_type']}，相关度{knowledge['relevance_score']:.2f}）；"
            
            # 合并所有文本特征，确保与测试上下文一致
            combined_text = f"{match_context}。{expert_analysis_text}。{top_knowledge_text}"
            text_data.append(combined_text)
        
        # 打印上下文匹配结果总结
        print(f"上下文匹配总结：找到 {found_count} 个，未找到 {not_found_count} 个")
        
        # 获取特征列
        feature_columns = match_processor.get_feature_columns()
        
        # 创建测试数据集
        test_dataset = MatchDataset(match_data, text_data, feature_columns)
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return test_loader
    
    def get_feature_info(self, data_path: str, batch_size: int = 100) -> Dict:
        """获取特征信息，支持批次处理
        
        Args:
            data_path: 数据路径
            batch_size: 批次大小，默认处理100个样本
            
        Returns:
            特征信息字典
        """
        from .match_data_processor import MatchDataProcessor
        
        match_processor = MatchDataProcessor(self.config)
        match_data = match_processor.process_data(data_path, batch_size=batch_size)
        
        return {
            "feature_columns": match_processor.get_feature_columns(),
            "label_column": match_processor.get_label_column(),
            "num_features": len(match_processor.get_feature_columns()),
            "num_classes": 3,  # win, draw, loss
            "num_samples": len(match_data)
        }