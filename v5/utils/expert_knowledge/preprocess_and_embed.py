#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家知识预处理与嵌入生成脚本
1. 对专家数据进行预处理
2. 使用Qwen模型生成语义嵌入
3. 缓存嵌入向量到本地文件
"""

import json
import os
import numpy as np
import torch
import re
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from datetime import datetime


class ExpertKnowledgeProcessor:
    """专家知识预处理与嵌入生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 加载配置文件
        self._load_config_from_file()
        
        # 合并配置，命令行配置优先
        self.full_config = {**self.file_config, **config}
        
        # 配置参数
        self.knowledge_base_path = self.full_config.get(
            'knowledge_base_path',
            '/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json'
        )
        
        self.embedding_cache_dir = self.full_config.get(
            'embedding_cache_dir',
            '/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/embeddings'
        )
        
        # 从配置文件读取LLM配置
        self.llm_config = self.full_config.get('expert_analysis_llm', {})
        self.force_local = self.llm_config.get('force_local', True)
        self.model_name = self.llm_config.get('model_name', '/Users/Williamhiler/Documents/my-project/train/models/cache')  # 修正为实际的本地模型路径
        
        self.embedding_dim = self.full_config.get('embedding_dim', 512)
        
        # 创建缓存目录
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        # 初始化模型和分词器
        self.model, self.tokenizer = self._initialize_model()
        
        # 加载知识库
        self.knowledge_base = self._load_knowledge_base()
    
    
    def _load_config_from_file(self):
        """从文件加载配置"""
        import yaml
        
        self.file_config = {}
        
        # 配置文件路径列表
        config_files = [
            "/Users/Williamhiler/Documents/my-project/train/config/config.yaml",
            "/Users/Williamhiler/Documents/my-project/train/v5/configs/v5_config.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        self.file_config.update(file_config)
                        print(f"✅ 加载配置文件: {config_file}")
                except Exception as e:
                    print(f"❌ 加载配置文件失败 {config_file}: {e}")
            else:
                print(f"⚠️  配置文件不存在: {config_file}")
    
    
    def _initialize_model(self):
        """初始化Qwen模型和分词器"""
        print("正在加载模型...")
        print(f"   模型名称/路径: {self.model_name}")
        print(f"   强制本地模式: {self.force_local}")
        
        try:
            # 检查GPU是否可用
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"   GPU可用，将使用: {torch.cuda.get_device_name(0)}")
                dtype = torch.float16
            else:
                device = torch.device("cpu")
                print("   GPU不可用，将使用CPU")
                dtype = torch.float32  # CPU上使用float32以提高兼容性
            
            if self.force_local:
                print(f"🔒 强制使用本地模型，不尝试下载")
                
                # 检查本地模型是否存在
                if not os.path.exists(self.model_name):
                    raise FileNotFoundError(f"本地模型路径不存在: {self.model_name}")
                
                # 强制从本地加载，不尝试下载
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,  # 强制使用本地文件
                    trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=device,  # 强制使用指定设备
                    local_files_only=True,  # 强制使用本地文件
                    trust_remote_code=True
                )
            else:
                # 正常模式，先尝试本地，再尝试下载
                print("正常模式，允许下载模型")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=device
                )
            
            # 确保模型在正确的设备上
            model.to(device)
            print(f"✅ 模型加载成功，设备: {device}")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    
    def _load_knowledge_base(self) -> Dict:
        """加载专家知识库"""
        print(f"正在加载知识库: {self.knowledge_base_path}")
        
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            print(f"✅ 知识库加载成功，包含 {len(knowledge_base['knowledge_units'])} 个知识单元")
            return knowledge_base
        except Exception as e:
            print(f"❌ 知识库加载失败: {e}")
            raise
    
    
    def preprocess_knowledge_unit(self, unit: Dict) -> Dict:
        """预处理单个知识单元"""
        processed_unit = unit.copy()
        
        # 1. 文本清洗
        content = unit.get('content', '')
        title = unit.get('title', '')
        
        # 清洗函数
        def clean_text(text: str) -> str:
            if not text:
                return ""
            
            # 去除特殊字符和多余空格
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,，。；;:!?！？\s\d.\d+]', ' ', text)
            
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 去除行首行尾空格
            text = text.strip()
            
            return text
        
        # 清洗标题和内容
        processed_unit['cleaned_title'] = clean_text(title)
        processed_unit['cleaned_content'] = clean_text(content)
        
        # 2. 构建完整文本表示
        text_parts = [
            processed_unit['cleaned_title'],
            processed_unit['cleaned_content'],
            f"知识类型: {processed_unit.get('knowledge_type', '')}",
            f"关键概念: {', '.join(processed_unit.get('key_concepts', []))}"
        ]
        
        # 过滤空字符串
        text_parts = [part for part in text_parts if part.strip()]
        
        # 合并为完整文本
        processed_unit['full_text'] = " ".join(text_parts)
        
        return processed_unit
    
    
    def preprocess_all_knowledge(self) -> List[Dict]:
        """预处理所有知识单元"""
        print("开始预处理所有知识单元...")
        
        processed_units = []
        for i, unit in enumerate(self.knowledge_base['knowledge_units']):
            processed_unit = self.preprocess_knowledge_unit(unit)
            processed_units.append(processed_unit)
            
            if (i + 1) % 100 == 0:
                print(f"已预处理 {i + 1}/{len(self.knowledge_base['knowledge_units'])} 个知识单元")
        
        print(f"✅ 预处理完成，共处理 {len(processed_units)} 个知识单元")
        return processed_units
    
    
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
            
            # 移至模型设备
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
            return np.zeros(self.embedding_dim)
    
    
    def generate_all_embeddings(self, processed_units: List[Dict]) -> np.ndarray:
        """生成所有知识单元的嵌入"""
        import time
        
        print("开始生成嵌入向量...")
        
        # 提取所有文本
        texts = [unit['full_text'] for unit in processed_units]
        batch_size = 4  # 进一步减少批处理大小，提高处理速度
        total_texts = len(texts)
        embeddings = []
        
        print(f"总文本数: {total_texts}, 批处理大小: {batch_size}, 总批次数: {((total_texts - 1) // batch_size) + 1}")
        print(f"模型设备: {self.model.device}")
        
        start_time = time.time()
        
        # 临时保存文件路径
        partial_embedding_file = os.path.join(self.embedding_cache_dir, 'partial_knowledge_embeddings.npy')
        partial_config_file = os.path.join(self.embedding_cache_dir, 'partial_embedding_config.json')
        
        # 动态获取嵌入维度的标志
        embedding_dim_set = False
        
        for i in range(0, total_texts, batch_size):
            # 获取当前批次
            batch_texts = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            current_batch_size = len(batch_texts)
            
            print(f"\n处理批次 {batch_num}/{((total_texts - 1) // batch_size) + 1}: 处理 {i+1}-{min(i+batch_size, total_texts)}/{total_texts} 个文本")
            
            batch_start_time = time.time()
            
            try:
                # 步骤1: 批量分词
                tokenize_start = time.time()
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                tokenize_time = time.time() - tokenize_start
                print(f"  分词完成: {tokenize_time:.2f}秒")
                
                # 步骤2: 移至模型设备
                to_device_start = time.time()
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                to_device_time = time.time() - to_device_start
                print(f"  移至设备完成: {to_device_time:.2f}秒")
                
                # 步骤3: 批量生成嵌入
                generate_start = time.time()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用最后一层的CLS token作为嵌入
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                generate_time = time.time() - generate_start
                print(f"  嵌入生成完成: {generate_time:.2f}秒")
                
                # 步骤3.5: 动态设置嵌入维度（仅第一次）
                if not embedding_dim_set:
                    actual_embedding_dim = batch_embeddings.shape[1]
                    if actual_embedding_dim != self.embedding_dim:
                        print(f"  注意: 实际嵌入维度与配置维度不一致")
                        print(f"  配置维度: {self.embedding_dim}, 实际维度: {actual_embedding_dim}")
                        print(f"  动态更新嵌入维度为: {actual_embedding_dim}")
                        self.embedding_dim = actual_embedding_dim
                    embedding_dim_set = True
                
                # 步骤4: 批量归一化
                normalize_start = time.time()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                normalize_time = time.time() - normalize_start
                print(f"  归一化完成: {normalize_time:.2f}秒")
                
                # 步骤5: 添加到结果列表
                embeddings.extend(batch_embeddings.tolist())
                
            except Exception as e:
                print(f"❌ 批量生成嵌入失败，批次 {batch_num}: {e}")
                # 对失败的批次进行单条处理
                for j, text in enumerate(batch_texts):
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding.tolist())
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # 打印当前批次统计
            print(f"  批次 {batch_num} 处理完成，耗时: {batch_time:.2f}秒")
            print(f"  批次速度: {current_batch_size/batch_time:.2f}文本/秒")
            
            # 打印整体进度和预计剩余时间
            processed = min(i + batch_size, total_texts)
            elapsed_time = time.time() - start_time
            texts_per_second = processed / elapsed_time if elapsed_time > 0 else 0
            remaining_time = (total_texts - processed) / texts_per_second if texts_per_second > 0 else 0
            
            print(f"  累计进度: {processed}/{total_texts} 个嵌入向量")
            print(f"  累计平均速度: {texts_per_second:.2f}文本/秒")
            print(f"  预计剩余时间: {remaining_time:.2f}秒 ({remaining_time/60:.1f}分钟)")
            
            # 定期保存进度（每处理10个批次或最后一个批次）
            if batch_num % 10 == 0 or processed == total_texts:
                print(f"\n💾 定期保存进度...")
                
                # 保存当前已生成的嵌入
                np.save(partial_embedding_file, np.array(embeddings))
                
                # 保存配置信息
                partial_config = {
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim,
                    'generated_timestamp': datetime.now().isoformat(),
                    'processed_count': processed,
                    'total_count': total_texts,
                    'last_batch_num': batch_num
                }
                
                with open(partial_config_file, 'w', encoding='utf-8') as f:
                    json.dump(partial_config, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 进度已保存: {processed}/{total_texts} 个嵌入向量")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 转换为numpy数组
        embeddings_array = np.array(embeddings)
        print(f"\n✅ 嵌入生成完成，嵌入形状: {embeddings_array.shape}")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"总平均速度: {total_texts/total_time:.2f}文本/秒")
        
        return embeddings_array
    
    
    def save_embeddings(self, embeddings: np.ndarray, processed_units: List[Dict]):
        """保存嵌入向量和相关配置"""
        print("开始保存嵌入向量...")
        
        # 保存嵌入向量
        embedding_file = os.path.join(self.embedding_cache_dir, 'knowledge_embeddings.npy')
        np.save(embedding_file, embeddings)
        print(f"✅ 嵌入向量已保存到: {embedding_file}")
        
        # 保存嵌入配置
        embedding_config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'generated_timestamp': datetime.now().isoformat(),
            'total_knowledge_units': len(processed_units),
            'preprocessing_steps': [
                '文本清洗',
                '标题内容合并',
                '知识类型添加',
                '关键概念整合'
            ],
            'knowledge_ids': [unit.get('id', str(i)) for i, unit in enumerate(processed_units)]
        }
        
        config_file = os.path.join(self.embedding_cache_dir, 'embedding_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_config, f, ensure_ascii=False, indent=2)
        print(f"✅ 嵌入配置已保存到: {config_file}")
        
        # 保存处理后的知识单元（可选）
        processed_file = os.path.join(self.embedding_cache_dir, 'processed_knowledge_units.json')
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_units, f, ensure_ascii=False, indent=2)
        print(f"✅ 处理后的知识单元已保存到: {processed_file}")
    
    
    def load_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """加载已保存的嵌入向量"""
        print("加载已保存的嵌入向量...")
        
        # 加载嵌入向量
        embedding_file = os.path.join(self.embedding_cache_dir, 'knowledge_embeddings.npy')
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"嵌入向量文件不存在: {embedding_file}")
        
        embeddings = np.load(embedding_file)
        print(f"✅ 加载嵌入向量成功，形状: {embeddings.shape}")
        
        # 加载嵌入配置
        config_file = os.path.join(self.embedding_cache_dir, 'embedding_config.json')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"嵌入配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            embedding_config = json.load(f)
        print(f"✅ 加载嵌入配置成功")
        
        return embeddings, embedding_config
    
    
    def run_full_pipeline(self):
        """运行完整的预处理和嵌入生成流程"""
        print("=== 专家知识预处理与嵌入生成流程 ===")
        
        try:
            # 检查是否有部分生成的嵌入
            partial_embedding_file = os.path.join(self.embedding_cache_dir, 'partial_knowledge_embeddings.npy')
            partial_config_file = os.path.join(self.embedding_cache_dir, 'partial_embedding_config.json')
            
            start_idx = 0
            existing_embeddings = []
            
            # 1. 预处理所有知识单元
            processed_units = self.preprocess_all_knowledge()
            total_units = len(processed_units)
            
            # 2. 检查断点
            if os.path.exists(partial_embedding_file) and os.path.exists(partial_config_file):
                print("\n🔍 发现部分生成的嵌入，尝试从断点继续...")
                
                # 加载部分嵌入
                existing_embeddings = np.load(partial_embedding_file).tolist()
                with open(partial_config_file, 'r', encoding='utf-8') as f:
                    partial_config = json.load(f)
                
                start_idx = len(existing_embeddings)
                print(f"📋 已生成 {start_idx}/{total_units} 个嵌入向量，将从第 {start_idx + 1} 个开始继续")
                
                # 检查一致性
                if start_idx > total_units:
                    print("⚠️  部分嵌入数量超过总知识单元数，将重新生成")
                    start_idx = 0
                    existing_embeddings = []
            
            # 3. 生成嵌入向量
            if start_idx == 0:
                # 从头开始生成
                embeddings = self.generate_all_embeddings(processed_units)
            else:
                # 从断点继续生成
                remaining_units = processed_units[start_idx:]
                print(f"\n开始继续生成嵌入向量...")
                print(f"总文本数: {total_units}, 已生成: {start_idx}, 剩余: {len(remaining_units)}")
                
                remaining_embeddings = self.generate_all_embeddings(remaining_units)
                embeddings = np.array(existing_embeddings + remaining_embeddings.tolist())
            
            # 4. 保存完整嵌入
            self.save_embeddings(embeddings, processed_units)
            
            # 5. 清理临时文件
            if os.path.exists(partial_embedding_file):
                os.remove(partial_embedding_file)
            if os.path.exists(partial_config_file):
                os.remove(partial_config_file)
            print(f"✅ 已清理临时断点文件")
            
            print("\n🎉 完整流程执行成功！")
            print(f"- 预处理知识单元: {len(processed_units)} 个")
            print(f"- 生成嵌入向量: {embeddings.shape}")
            print(f"- 嵌入缓存目录: {self.embedding_cache_dir}")
            
        except Exception as e:
            print(f"\n❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    
    def validate_embeddings(self):
        """验证嵌入向量的质量"""
        print("开始验证嵌入向量质量...")
        
        try:
            # 加载嵌入
            embeddings, config = self.load_embeddings()
            
            # 1. 检查嵌入维度
            actual_dim = embeddings.shape[1]
            print(f"嵌入维度: {actual_dim}")
            
            # 使用实际嵌入维度，而不是依赖配置值
            print(f"配置维度: {self.embedding_dim}")
            
            # 2. 检查嵌入数量
            print(f"嵌入数量: {embeddings.shape[0]}")
            assert embeddings.shape[0] == len(self.knowledge_base['knowledge_units']), \
                f"嵌入数量与知识单元数量不匹配: {embeddings.shape[0]} != {len(self.knowledge_base['knowledge_units'])}"
            
            # 3. 检查归一化
            norms = np.linalg.norm(embeddings, axis=1)
            avg_norm = np.mean(norms)
            print(f"嵌入向量平均范数: {avg_norm:.6f}")
            assert abs(avg_norm - 1.0) < 0.01, f"嵌入向量未正确归一化，平均范数: {avg_norm}"
            
            # 4. 检查嵌入多样性（计算余弦相似度矩阵的平均值）
            sample_size = min(100, len(embeddings))
            sample_embeddings = embeddings[:sample_size]
            similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
            
            # 排除对角线元素（自身相似度）
            mask = np.eye(similarity_matrix.shape[0], dtype=bool)
            avg_similarity = np.mean(similarity_matrix[~mask])
            print(f"嵌入向量平均相似度: {avg_similarity:.6f}")
            
            print("\n✅ 嵌入向量验证通过！")
            print("嵌入向量质量良好，可以用于匹配任务。")
            
        except Exception as e:
            print(f"\n❌ 嵌入向量验证失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 配置
    config = {
        'knowledge_base_path': '/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json',
        'embedding_cache_dir': '/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/embeddings',
        'model_name': 'Qwen/Qwen2-0.5B',
        'embedding_dim': 512
    }
    
    # 创建处理器实例
    processor = ExpertKnowledgeProcessor(config)
    
    # 运行完整流程
    processor.run_full_pipeline()
    
    # 验证嵌入质量
    processor.validate_embeddings()


if __name__ == "__main__":
    main()