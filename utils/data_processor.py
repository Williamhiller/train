import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple


class SoccerDataProcessor:
    """
    足球比赛数据处理器，用于加载、预处理和转换足球比赛数据
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self._fitted = False
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSON格式的足球比赛数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []
    
    def preprocess_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        预处理足球比赛数据
        
        Args:
            raw_data: 原始数据列表
            
        Returns:
            预处理后的DataFrame
        """
        processed_data = []
        
        for item in raw_data:
            processed_item = self._process_single_item(item)
            processed_data.append(processed_item)
        
        df = pd.DataFrame(processed_data)
        return df
    
    def _process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项
        """
        result = {}
        
        # 基础信息
        if 'input' in item:
            input_data = item['input']
            # 赔率信息
            result['home_odds'] = input_data.get('home_odds', 0)
            result['draw_odds'] = input_data.get('draw_odds', 0)
            result['away_odds'] = input_data.get('away_odds', 0)
            
            # 排名信息
            result['home_ranking'] = input_data.get('home_ranking', 0)
            result['away_ranking'] = input_data.get('away_ranking', 0)
            result['ranking_diff'] = result['away_ranking'] - result['home_ranking']
            
            # 近期战绩统计
            home_results = input_data.get('home_recent_results', [])
            away_results = input_data.get('away_recent_results', [])
            
            result['home_win_rate'] = self._calculate_win_rate(home_results)
            result['away_win_rate'] = self._calculate_win_rate(away_results)
            result['home_goals_scored_avg'] = input_data.get('home_goals_scored', 0) / 5 if home_results else 0
            result['home_goals_conceded_avg'] = input_data.get('home_goals_conceded', 0) / 5 if home_results else 0
            result['away_goals_scored_avg'] = input_data.get('away_goals_scored', 0) / 5 if away_results else 0
            result['away_goals_conceded_avg'] = input_data.get('away_goals_conceded', 0) / 5 if away_results else 0
            
            # 积分信息
            result['home_points'] = input_data.get('home_points', 0)
            result['away_points'] = input_data.get('away_points', 0)
            result['points_diff'] = result['home_points'] - result['away_points']
            
            # 历史交锋
            head_to_head = input_data.get('head_to_head', [])
            result['home_h2h_win_rate'] = self._calculate_h2h_win_rate(head_to_head, 'H')
            result['away_h2h_win_rate'] = self._calculate_h2h_win_rate(head_to_head, 'A')
            
            # 状态值
            result['home_form'] = input_data.get('home_form', 0.5)
            result['away_form'] = input_data.get('away_form', 0.5)
            
        # 目标输出
        if 'output' in item:
            output_data = item['output']
            result['result'] = output_data.get('result', 'D')
            result['score'] = output_data.get('score', '0-0')
            
            # 解析比分
            if result['score'] and '-' in result['score']:
                try:
                    home_score, away_score = map(int, result['score'].split('-'))
                    result['home_goals'] = home_score
                    result['away_goals'] = away_score
                except:
                    result['home_goals'] = 0
                    result['away_goals'] = 0
            else:
                result['home_goals'] = 0
                result['away_goals'] = 0
        
        return result
    
    def _calculate_win_rate(self, results: List[str]) -> float:
        """
        计算胜率
        """
        if not results:
            return 0.5
        wins = sum(1 for r in results if r == 'W')
        return wins / len(results)
    
    def _calculate_h2h_win_rate(self, results: List[str], team: str) -> float:
        """
        计算历史交锋胜率
        """
        if not results:
            return 0.33
        wins = sum(1 for r in results if r == team)
        return wins / len(results)
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        准备特征矩阵和标签
        
        Args:
            df: 预处理后的DataFrame
            is_training: 是否为训练数据（包含标签）
            
        Returns:
            (特征矩阵, 标签向量)，如果不是训练数据则标签为None
        """
        # 定义特征列
        feature_cols = [
            'home_odds', 'draw_odds', 'away_odds',
            'home_ranking', 'away_ranking', 'ranking_diff',
            'home_win_rate', 'away_win_rate',
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'home_points', 'away_points', 'points_diff',
            'home_h2h_win_rate', 'away_h2h_win_rate',
            'home_form', 'away_form'
        ]
        
        # 确保所有特征列都存在
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        
        # 标准化特征
        if is_training:
            X = self.scaler.fit_transform(X)
            self._fitted = True
        else:
            if self._fitted:
                X = self.scaler.transform(X)
            else:
                print("警告: 特征标准化器尚未拟合")
        
        # 准备标签
        y = None
        if is_training and 'result' in df.columns:
            # 将结果转换为数值标签
            result_map = {'W': 0, 'D': 1, 'L': 2}
            y = df['result'].map(result_map).fillna(1).values
        
        return X, y
    
    def convert_to_model_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        将数据转换为模型训练所需的格式（文本对）
        
        Args:
            data: 原始数据列表
            
        Returns:
            模型训练格式的数据列表
        """
        formatted_data = []
        
        for item in data:
            input_text = self._create_input_prompt(item)
            output_text = self._create_output_text(item)
            
            formatted_data.append({
                'input': input_text,
                'output': output_text
            })
        
        return formatted_data
    
    def _create_input_prompt(self, item: Dict[str, Any]) -> str:
        """
        创建输入提示文本
        """
        input_data = item.get('input', {})
        
        prompt = f"""
        预测以下足球比赛的结果：
        主队：{input_data.get('home_team', '未知')}
        客队：{input_data.get('away_team', '未知')}
        联赛：{input_data.get('league', '未知')}
        主队赔率：{input_data.get('home_odds', 0)}
        平局赔率：{input_data.get('draw_odds', 0)}
        客队赔率：{input_data.get('away_odds', 0)}
        主队排名：{input_data.get('home_ranking', 0)}
        客队排名：{input_data.get('away_ranking', 0)}
        主队近期战绩：{', '.join(input_data.get('home_recent_results', []))}
        客队近期战绩：{', '.join(input_data.get('away_recent_results', []))}
        历史交锋：{', '.join(input_data.get('head_to_head', []))}
        """
        
        return prompt.strip()
    
    def _create_output_text(self, item: Dict[str, Any]) -> str:
        """
        创建输出文本
        """
        output_data = item.get('output', {})
        result = output_data.get('result', 'D')
        score = output_data.get('score', '0-0')
        
        result_map = {'W': '主队胜', 'D': '平局', 'L': '客队胜'}
        result_text = result_map.get(result, '未知')
        
        return f"结果: {result_text}\n比分: {score}"
    
    def save_processor(self, save_path: str):
        """
        保存数据处理器的状态
        """
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            '_fitted': self._fitted
        }, save_path)
    
    def load_processor(self, load_path: str):
        """
        加载数据处理器的状态
        """
        import joblib
        data = joblib.load(load_path)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self._fitted = data['_fitted']


def prepare_training_data(raw_data: List[Dict[str, Any]], output_file: str):
    """
    准备训练数据并保存
    
    Args:
        raw_data: 原始数据
        output_file: 输出文件路径
    """
    processor = SoccerDataProcessor()
    formatted_data = processor.convert_to_model_format(raw_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练数据已保存至: {output_file}")


def preprocess_and_split_data(
    file_path: str,
    train_output: str = './data/train.json',
    val_output: str = './data/validation.json',
    test_output: str = './data/test.json',
    test_ratio: float = 0.1,
    val_ratio: float = 0.1
):
    """
    预处理数据并分割为训练集、验证集和测试集
    
    Args:
        file_path: 原始数据文件路径
        train_output: 训练集输出路径
        val_output: 验证集输出路径
        test_output: 测试集输出路径
        test_ratio: 测试集比例
        val_ratio: 验证集比例
    """
    processor = SoccerDataProcessor()
    data = processor.load_data(file_path)
    
    # 随机打乱数据
    import random
    random.shuffle(data)
    
    # 分割数据
    total = len(data)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    
    test_data = data[:test_size]
    val_data = data[test_size:test_size + val_size]
    train_data = data[test_size + val_size:]
    
    # 保存数据
    for data_split, output_path in [(train_data, train_output), (val_data, val_output), (test_data, test_output)]:
        prepare_training_data(data_split, output_path)
    
    print(f"数据分割完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")