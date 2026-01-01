import json
import re
import os
from collections import defaultdict

# 输入输出文件路径
input_file = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_data/expert_training_data.json"
output_file = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_data/cleaned_expert_data.json"
qwen_finetune_file = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_data/qwen_finetune_data.json"

# 读取原始数据
def read_raw_data():
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# 清理文本内容
def clean_text(text):
    # 去除特殊字符和乱码
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 去除所有非中文字符和基本标点符号
    text = re.sub(r'[^\u4e00-\u9fa5。，！？；：、（）《》【】]', '', text)
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 去除PDF页眉页脚
    text = re.sub(r'六步解欧赔', '', text)
    text = re.sub(r'闲说310', '', text)
    # 去除重复的标点符号
    text = re.sub(r'([。，！？；：])\1+', r'\1', text)
    # 去除首尾空格
    text = text.strip()
    return text

# 构建有效的问答对
def build_qa_pairs(raw_data):
    qa_pairs = []
    
    # 统计重复内容，用于去重
    content_seen = set()
    
    for item in raw_data:
        # 清理prompt和response
        raw_prompt = item.get('prompt', '')
        raw_response = item.get('response', '')
        
        # 跳过无效数据
        if not raw_prompt or not raw_response:
            continue
        
        # 清理文本
        cleaned_prompt = clean_text(raw_prompt)
        cleaned_response = clean_text(raw_response)
        
        # 跳过无效内容
        if not cleaned_prompt or not cleaned_response or len(cleaned_response) < 50:
            continue
        
        # 直接使用清理后的回答，稍后在optimize_qa_pairs中生成多样化问题
        answer = cleaned_response
        
        # 去重检查：使用回答内容作为唯一标识
        content_key = answer[:200]  # 使用前200个字符作为去重依据
        if content_key not in content_seen:
            content_seen.add(content_key)
            # 暂时使用空问题，后续在optimize_qa_pairs中生成
            qa_pairs.append({
                "question": "",
                "answer": answer
            })
    
    return qa_pairs

# 进一步优化问答对，提高问题针对性和多样性
def optimize_qa_pairs(qa_pairs):
    optimized_pairs = []
    
    # 定义丰富的主题关键词和对应的多样化问题模板
    topic_patterns = [
        # 核心概念类
        (r'平衡|失衡|均衡|分流', [
            "请解释足球比赛欧赔中的平衡与失衡法则",
            "什么是欧赔的平衡与失衡法则？",
            "如何理解足球赔率的平衡与失衡？",
            "欧赔平衡法则在比赛预测中的应用是什么？",
            "足球赔率中的平衡状态是如何维持的？",
            "失衡的赔率通常意味着什么？"
        ]),
        # 冷门分析类
        (r'冷门|冷盘|大冷|中冷|温0|冷0', [
            "请解释足球比赛中的冷门分析方法",
            "如何分析足球比赛中的冷门情况？",
            "足球比赛中冷门的赔率特征是什么？",
            "庄家如何处理冷门比赛的赔率？",
            "冷门比赛的赔率变化有哪些规律？",
            "如何识别可能出现冷门的比赛？"
        ]),
        # 庄家策略类
        (r'庄家|菠菜|筹码|赔率调整|变盘|操盘', [
            "请解释庄家的赔率策略和操盘手法",
            "庄家是如何制定赔率策略的？",
            "庄家的操盘手法有哪些？",
            "筹码分布如何影响庄家的赔率调整？",
            "庄家变盘的动机和动力是什么？",
            "主流庄家和小庄家的开盘策略有什么不同？"
        ]),
        # 赔率分析类
        (r'欧赔|赔率|盘口|陪率|胜赔|平赔|负赔', [
            "请解释足球比赛的欧赔分析方法",
            "如何分析足球比赛的欧赔？",
            "足球赔率分析的关键方法是什么？",
            "盘口分析在足球预测中的作用是什么？",
            "胜赔、平赔、负赔之间的关系是什么？",
            "如何通过赔率组合判断比赛结果？"
        ]),
        # 预测方法类
        (r'预测|分析|判断|方法|技巧|思路', [
            "请解释足球比赛预测的关键规则和方法",
            "足球比赛预测有哪些关键规则？",
            "如何提高足球比赛预测的准确性？",
            "足球预测的主要分析方法有哪些？",
            "足球比赛预测需要考虑哪些因素？",
            "如何结合多种方法进行足球预测？"
        ]),
        # 六步解欧赔类
        (r'六步|步骤|方法|流程|悟|度|识|借|利|式', [
            "请解释六步解欧赔的具体步骤和方法",
            "六步解欧赔的具体步骤是什么？",
            "如何使用六步解欧赔方法分析比赛？",
            "六步解欧赔的核心方法是什么？",
            "六步解欧赔中的'悟'、'度'、'识'分别指什么？",
            "六步解欧赔方法的优势和局限性是什么？"
        ]),
        # 平局分析类
        (r'平局|平赔|出1|1的手法', [
            "请解释足球比赛中的平局分析方法",
            "如何分析足球比赛的平局可能性？",
            "平赔在足球赔率中的作用是什么？",
            "平局的赔率特征有哪些？",
            "庄家开出平局的常见手法有哪些？",
            "如何通过赔率判断平局的可能性？"
        ]),
        # 胜负分析类
        (r'胜赔|负赔|胜负|出3|出0|0的手法', [
            "请解释足球比赛中的胜负赔率分析",
            "如何分析足球比赛的胜负可能性？",
            "胜赔和负赔的变化规律是什么？",
            "胜负赔率在预测中的应用技巧有哪些？",
            "庄家开出主胜的常见手法有哪些？",
            "客胜的赔率特征是什么？"
        ]),
        # 赛季特征类
        (r'赛季|战意|排名|联赛|杯赛|友谊赛', [
            "请解释赛季初和赛季末的赔率特点",
            "球队战意如何影响赔率？",
            "排名对赔率有什么影响？",
            "如何分析不同赛季阶段的赔率？",
            "联赛和杯赛的赔率制定有什么不同？",
            "不同联赛的赔率特点有哪些？"
        ]),
        # 基本面分析类
        (r'基本面|近况|往绩|实力|状态|伤病', [
            "基本面在足球赔率分析中的作用是什么？",
            "如何结合球队近况分析赔率？",
            "往绩对赔率的影响有多大？",
            "基本面分析的关键要素有哪些？",
            "如何平衡基本面和赔率分析？",
            "球队状态对赔率的影响是什么？"
        ]),
        # 博彩心理类
        (r'心理|心态|贪婪|恐惧|热|诱|阻', [
            "博彩中的心理因素有什么影响？",
            "如何克服博彩中的贪婪和恐惧？",
            "庄家如何利用心理因素制定赔率？",
            "热门比赛的赔率特征是什么？",
            "如何避免被赔率诱导？",
            "博彩中的心态管理重要吗？"
        ]),
        # 复盘总结类
        (r'复盘|再认识|经验|教训|总结', [
            "为什么复盘在博彩中很重要？",
            "如何进行有效的比赛复盘？",
            "复盘能带来哪些好处？",
            "职业博彩者是如何复盘的？",
            "复盘时需要关注哪些要点？",
            "如何从复盘中学习提高？"
        ]),
        # 合作与借势类
        (r'借|合作|共享|互纠|团队', [
            "博彩中的'借'是什么意思？",
            "如何在博彩中有效借势？",
            "团队合作对博彩有什么帮助？",
            "为什么说'剩者为王'是博彩的生存法则？",
            "如何实现博彩中的资源共享？",
            "合作博彩的优势是什么？"
        ]),
        # 赔率变动类
        (r'变盘|赔率调整|开赔时间|陪付率', [
            "庄家为什么会变盘？",
            "赔率变动的原因有哪些？",
            "开赔时间对赔率有什么影响？",
            "不同庄家的变盘策略有什么不同？",
            "陪付率的高低意味着什么？",
            "如何解读赔率的大幅变动？"
        ]),
        # 威廉希尔等特定庄家类
        (r'威廉|立博|伟德|Interwellen|SSP', [
            "威廉希尔的赔率特点是什么？",
            "立博与威廉希尔的赔率有什么不同？",
            "主流庄家的赔率制定有什么特点？",
            "为什么威廉希尔的平赔普遍偏低？",
            "小庄家的赔率有什么参考价值？",
            "如何利用不同庄家的赔率差异？"
        ])
    ]
    
    # 统计优化后的内容，避免重复
    optimized_seen = set()
    
    for pair in qa_pairs:
        answer = pair["answer"]
        
        # 寻找所有匹配的主题
        matching_templates = []
        for pattern, templates in topic_patterns:
            if re.search(pattern, answer):
                matching_templates.extend(templates)
        
        # 如果没有匹配到主题，使用通用问题
        if not matching_templates:
            matching_templates = [
                "请解释足球比赛预测的相关知识",
                "足球比赛预测的关键要素有哪些？",
                "如何分析足球比赛？",
                "足球比赛预测的核心方法是什么？",
                "如何提高足球比赛预测的准确性？",
                "足球赔率分析的基本思路是什么？"
            ]
        
        # 随机选择多个问题模板，增加多样性
        import random
        num_questions = 2  # 每个回答生成2个不同的问题
        selected_questions = random.sample(matching_templates, min(num_questions, len(matching_templates)))
        
        # 去重检查
        content_key = answer[:150]  # 只使用回答内容作为去重依据，允许不同问题对应相同回答
        if content_key not in optimized_seen:
            optimized_seen.add(content_key)
            # 为每个回答生成多个不同的问题
            for question in selected_questions:
                optimized_pairs.append({
                    "question": question,
                    "answer": answer
                })
    
    return optimized_pairs

# 将QA数据转换为Qwen微调所需的格式
def convert_to_qwen_format(qa_pairs):
    """将QA对转换为Qwen模型微调所需的instruction-input-output格式"""
    qwen_data = []
    for pair in qa_pairs:
        # 对于纯问答对，instruction设为问题，input为空，output设为回答
        qwen_data.append({
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"]
        })
    return qwen_data

# 主函数
def main():
    print("开始清洗专家数据...")
    
    # 读取原始数据
    raw_data = read_raw_data()
    print(f"原始数据条数: {len(raw_data)}")
    
    # 构建问答对
    qa_pairs = build_qa_pairs(raw_data)
    print(f"构建问答对后: {len(qa_pairs)}条")
    
    # 优化问答对
    optimized_pairs = optimize_qa_pairs(qa_pairs)
    print(f"优化后: {len(optimized_pairs)}条")
    
    # 输出清洗后的数据（通用QA格式）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(optimized_pairs, f, ensure_ascii=False, indent=2)
    
    # 转换为Qwen微调格式并输出
    qwen_data = convert_to_qwen_format(optimized_pairs)
    with open(qwen_finetune_file, 'w', encoding='utf-8') as f:
        json.dump(qwen_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据清洗完成！")
    print(f"通用QA格式数据已保存到: {output_file}")
    print(f"Qwen微调格式数据已保存到: {qwen_finetune_file}")
    print(f"清洗前后对比: {len(raw_data)} → {len(optimized_pairs)}条")
    
    # 输出前5条作为示例
    print("\n示例数据:")
    for i, pair in enumerate(optimized_pairs[:5]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"问题: {pair['question']}")
        print(f"回答: {pair['answer'][:100]}...")

if __name__ == "__main__":
    main()
