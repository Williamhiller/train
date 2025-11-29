import json
from typing import Dict, Any


def prepare_prediction_input(input_data: Dict[str, Any]) -> str:
    """
    准备预测输入的提示文本
    
    Args:
        input_data: 包含比赛信息的字典
        
    Returns:
        格式化的提示文本
    """
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
    主队近期战绩：{', '.join(map(str, input_data.get('home_recent_results', [])))}
    客队近期战绩：{', '.join(map(str, input_data.get('away_recent_results', [])))}
    历史交锋：{', '.join(map(str, input_data.get('head_to_head', [])))}
    """
    
    # 添加可选信息
    if 'home_goals_scored' in input_data:
        prompt += f"主队进球数（近5场）：{input_data['home_goals_scored']}\n"
    if 'away_goals_scored' in input_data:
        prompt += f"客队进球数（近5场）：{input_data['away_goals_scored']}\n"
    if 'home_points' in input_data:
        prompt += f"主队积分：{input_data['home_points']}\n"
    if 'away_points' in input_data:
        prompt += f"客队积分：{input_data['away_points']}\n"
    
    return prompt.strip()


def parse_prediction_output(output_text: str) -> Dict[str, Any]:
    """
    解析模型预测输出
    
    Args:
        output_text: 模型输出的文本
        
    Returns:
        解析后的预测结果
    """
    result = {
        'prediction': 'D',
        'confidence': 0.5,
        'detailed_probabilities': {
            'W': 0.33,
            'D': 0.33,
            'L': 0.33
        },
        'score': None
    }
    
    # 解析结果
    lines = output_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '结果:' in line:
            result_text = line.split('结果:')[-1].strip()
            if '主队胜' in result_text or 'W' in result_text:
                result['prediction'] = 'W'
                result['detailed_probabilities'] = {'W': 0.7, 'D': 0.2, 'L': 0.1}
                result['confidence'] = 0.7
            elif '客队胜' in result_text or 'L' in result_text:
                result['prediction'] = 'L'
                result['detailed_probabilities'] = {'W': 0.1, 'D': 0.2, 'L': 0.7}
                result['confidence'] = 0.7
            else:
                result['prediction'] = 'D'
                result['detailed_probabilities'] = {'W': 0.2, 'D': 0.6, 'L': 0.2}
                result['confidence'] = 0.6
        
        # 解析比分
        if '比分:' in line:
            score_text = line.split('比分:')[-1].strip()
            if '-' in score_text and score_text.replace('-', '').isdigit():
                result['score'] = score_text
    
    return result


def load_prediction_input(file_path: str) -> Dict[str, Any]:
    """
    加载预测输入文件
    
    Args:
        file_path: 输入文件路径
        
    Returns:
        输入数据字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载输入文件失败: {e}")
        return {}


def save_prediction_result(result: Dict[str, Any], output_file: str):
    """
    保存预测结果
    
    Args:
        result: 预测结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存至: {output_file}")


def validate_prediction_input(input_data: Dict[str, Any]) -> bool:
    """
    验证预测输入数据的有效性
    
    Args:
        input_data: 输入数据
        
    Returns:
        是否有效
    """
    required_fields = ['home_team', 'away_team', 'home_odds', 'draw_odds', 'away_odds']
    
    # 检查必填字段
    for field in required_fields:
        if field not in input_data:
            print(f"错误：缺少必填字段 {field}")
            return False
    
    # 验证赔率是否为有效数字
    odds_fields = ['home_odds', 'draw_odds', 'away_odds']
    for field in odds_fields:
        if not isinstance(input_data[field], (int, float)) or input_data[field] <= 0:
            print(f"错误：{field} 必须是正数")
            return False
    
    return True