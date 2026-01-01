#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专家知识编辑接口
用于手动调整和优化预处理后的专家知识
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


class ExpertKnowledgeEditor:
    """专家知识编辑器"""
    
    def __init__(self, knowledge_base_path: str = None):
        if knowledge_base_path is None:
            self.knowledge_base_path = "/Users/Williamhiler/Documents/my-project/train/v5/data/expert_knowledge/expert_knowledge_base.json"
        else:
            self.knowledge_base_path = knowledge_base_path
        
        self.backup_dir = os.path.join(os.path.dirname(self.knowledge_base_path), "backups")
        self.knowledge_base = None
        self.current_unit_index = 0
        
        # 创建备份目录
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 加载知识库
        self.load_knowledge_base()
    
    
    def load_knowledge_base(self):
        """加载知识库"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"✅ 成功加载知识库: {len(self.knowledge_base['knowledge_units'])} 个知识单元")
        except Exception as e:
            print(f"❌ 加载知识库失败: {e}")
            self.knowledge_base = None
    
    
    def save_knowledge_base(self, backup: bool = True):
        """保存知识库"""
        if backup:
            self.create_backup()
        
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            print("✅ 知识库保存成功")
        except Exception as e:
            print(f"❌ 保存知识库失败: {e}")
    
    
    def create_backup(self):
        """创建备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"knowledge_base_backup_{timestamp}.json")
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            print(f"✅ 备份已创建: {backup_file}")
        except Exception as e:
            print(f"❌ 备份创建失败: {e}")
    
    
    def list_knowledge_units(self, category: str = None, limit: int = 10, offset: int = 0):
        """列出知识单元"""
        units = self.knowledge_base["knowledge_units"]
        
        if category:
            category_key = self._get_category_key(category)
            if category_key in self.knowledge_base["category_index"]:
                indices = self.knowledge_base["category_index"][category_key]
                units = [units[i] for i in indices]
            else:
                print(f"❌ 未找到分类: {category}")
                return []
        
        # 分页显示
        total = len(units)
        start = offset
        end = min(offset + limit, total)
        
        print(f"\n=== 知识单元列表 ({start+1}-{end}/{total}) ===")
        
        for i, unit in enumerate(units[start:end], start+1):
            print(f"\n[{i}] 来源: {unit['source_document']} (第{unit['page_number']}页)")
            print(f"    类型: {unit['knowledge_type']}")
            print(f"    标题: {unit.get('title', '无标题')[:50]}...")
            print(f"    内容: {unit['content'][:100]}...")
            print(f"    概念: {', '.join(unit.get('key_concepts', [])[:5])}")
            print(f"    实用价值: {unit.get('practical_value', {})}")
        
        return units[start:end]
    
    
    def edit_knowledge_unit(self, unit_index: int):
        """编辑单个知识单元"""
        if unit_index < 0 or unit_index >= len(self.knowledge_base["knowledge_units"]):
            print(f"❌ 无效的知识单元索引: {unit_index}")
            return
        
        unit = self.knowledge_base["knowledge_units"][unit_index]
        
        print(f"\n=== 编辑知识单元 [{unit_index}] ===")
        print(f"来源文档: {unit['source_document']}")
        print(f"页面: {unit['page_number']}")
        print(f"标题: {unit.get('title', '无标题')}")
        print(f"当前类型: {unit['knowledge_type']}")
        print(f"内容预览: {unit['content'][:200]}...")
        
        # 显示当前信息
        print("\n当前信息:")
        print(f"1. 知识类型: {unit['knowledge_type']}")
        print(f"2. 标题: {unit.get('title', '')}")
        print(f"3. 关键概念: {', '.join(unit.get('key_concepts', []))}")
        print(f"4. 实用价值: {unit.get('practical_value', {})}")
        
        # 交互式编辑
        while True:
            print("\n编辑选项:")
            print("1. 修改知识类型")
            print("2. 修改标题")
            print("3. 修改关键概念")
            print("4. 修改实用价值评分")
            print("5. 查看完整内容")
            print("6. 保存并退出")
            print("7. 放弃修改")
            
            choice = input("请选择操作 (1-7): ").strip()
            
            if choice == "1":
                self._edit_knowledge_type(unit)
            elif choice == "2":
                self._edit_title(unit)
            elif choice == "3":
                self._edit_key_concepts(unit)
            elif choice == "4":
                self._edit_practical_value(unit)
            elif choice == "5":
                print(f"\n完整内容:\n{unit['content']}")
            elif choice == "6":
                self.save_knowledge_base()
                break
            elif choice == "7":
                print("放弃修改")
                break
            else:
                print("❌ 无效选择")
    
    
    def _edit_knowledge_type(self, unit: Dict):
        """编辑知识类型"""
        print("\n可用的知识类型:")
        types = [
            "philosophy", "fundamentals", "techniques", 
            "patterns", "practical", "psychology", "risk_management"
        ]
        
        for i, t in enumerate(types, 1):
            print(f"{i}. {t}")
        
        choice = input(f"选择新的知识类型 (当前: {unit['knowledge_type']}): ").strip()
        
        try:
            new_type = types[int(choice) - 1]
            old_type = unit['knowledge_type']
            unit['knowledge_type'] = new_type
            
            # 更新分类索引
            self._update_category_index(unit, old_type, new_type)
            
            print(f"✅ 知识类型已更新: {old_type} → {new_type}")
        except (ValueError, IndexError):
            print("❌ 无效选择")
    
    
    def _edit_title(self, unit: Dict):
        """编辑标题"""
        current_title = unit.get('title', '')
        new_title = input(f"输入新标题 (当前: {current_title}): ").strip()
        
        if new_title:
            unit['title'] = new_title
            print(f"✅ 标题已更新")
        else:
            print("标题未修改")
    
    
    def _edit_key_concepts(self, unit: Dict):
        """编辑关键概念"""
        current_concepts = unit.get('key_concepts', [])
        print(f"当前关键概念: {', '.join(current_concepts)}")
        
        new_concepts = input("输入新的关键概念（用逗号分隔）: ").strip()
        
        if new_concepts:
            unit['key_concepts'] = [c.strip() for c in new_concepts.split(',') if c.strip()]
            print(f"✅ 关键概念已更新")
            
            # 更新概念索引
            self._update_concept_index(unit)
        else:
            print("关键概念未修改")
    
    
    def _edit_practical_value(self, unit: Dict):
        """编辑实用价值"""
        current_value = unit.get('practical_value', {})
        
        print("当前实用价值评分:")
        for metric, score in current_value.items():
            print(f"  {metric}: {score:.3f}")
        
        print("\n修改评分 (0.0-1.0):")
        for metric in ["actionability", "specificity", "uniqueness", "clarity"]:
            current_score = current_value.get(metric, 0.0)
            new_score = input(f"  {metric} (当前: {current_score:.3f}): ").strip()
            
            try:
                new_score_float = float(new_score)
                if 0.0 <= new_score_float <= 1.0:
                    current_value[metric] = new_score_float
                else:
                    print(f"❌ 评分必须在0.0-1.0之间")
            except ValueError:
                print(f"❌ 无效输入，使用原值: {current_score:.3f}")
        
        unit['practical_value'] = current_value
        print("✅ 实用价值评分已更新")
    
    
    def _update_category_index(self, unit: Dict, old_type: str, new_type: str):
        """更新分类索引"""
        # 从旧分类中移除
        if old_type in self.knowledge_base["category_index"]:
            unit_index = self.knowledge_base["knowledge_units"].index(unit)
            if unit_index in self.knowledge_base["category_index"][old_type]:
                self.knowledge_base["category_index"][old_type].remove(unit_index)
        
        # 添加到新分类
        if new_type not in self.knowledge_base["category_index"]:
            self.knowledge_base["category_index"][new_type] = []
        
        unit_index = self.knowledge_base["knowledge_units"].index(unit)
        if unit_index not in self.knowledge_base["category_index"][new_type]:
            self.knowledge_base["category_index"][new_type].append(unit_index)
    
    
    def _update_concept_index(self, unit: Dict):
        """更新概念索引"""
        # 重新构建概念索引（简化实现）
        new_concept_index = {}
        
        for i, u in enumerate(self.knowledge_base["knowledge_units"]):
            concepts = u.get("key_concepts", [])
            for concept in concepts:
                if concept not in new_concept_index:
                    new_concept_index[concept] = []
                new_concept_index[concept].append(i)
        
        self.knowledge_base["concept_index"] = new_concept_index
    
    
    def _get_category_key(self, category_name: str) -> str:
        """获取分类键"""
        # 反向查找分类键
        category_mapping = {
            "赔率的哲学思维和认知方法": "philosophy",
            "赔率分析的基础理论": "fundamentals", 
            "具体的分析技巧和判断标准": "techniques",
            "常见的赔率模式和异常情况": "patterns",
            "实战应用和案例分析": "practical",
            "心理分析和行为模式": "psychology",
            "风险控制和资金管理": "risk_management"
        }
        
        return category_mapping.get(category_name, category_name)
    
    
    def batch_edit_by_category(self, category: str):
        """批量编辑某个分类的知识单元"""
        category_key = self._get_category_key(category)
        
        if category_key not in self.knowledge_base["category_index"]:
            print(f"❌ 未找到分类: {category}")
            return
        
        indices = self.knowledge_base["category_index"][category_key]
        print(f"\n=== 批量编辑分类: {category} ({len(indices)}个单元) ===")
        
        # 显示该分类下的知识单元
        units = [self.knowledge_base["knowledge_units"][i] for i in indices[:10]]  # 显示前10个
        
        print("样本预览:")
        for i, unit in enumerate(units, 1):
            print(f"{i}. {unit.get('title', '无标题')[:50]}...")
            print(f"   内容: {unit['content'][:80]}...")
        
        print(f"\n共找到 {len(indices)} 个知识单元")
        
        action = input("选择操作: 1.修改分类 2.添加概念 3.调整评分 4.退出: ").strip()
        
        if action == "1":
            self._batch_change_category(indices)
        elif action == "2":
            self._batch_add_concepts(indices)
        elif action == "3":
            self._batch_adjust_scores(indices)
        else:
            print("退出批量编辑")
    
    
    def _batch_change_category(self, indices: List[int]):
        """批量修改分类"""
        print("\n选择新的分类:")
        types = [
            "philosophy", "fundamentals", "techniques", 
            "patterns", "practical", "psychology", "risk_management"
        ]
        
        for i, t in enumerate(types, 1):
            print(f"{i}. {t}")
        
        choice = input("选择新分类: ").strip()
        
        try:
            new_type = types[int(choice) - 1]
            
            confirm = input(f"确认将 {len(indices)} 个单元修改为 {new_type} 分类? (y/n): ").strip().lower()
            
            if confirm == 'y':
                for idx in indices:
                    unit = self.knowledge_base["knowledge_units"][idx]
                    old_type = unit['knowledge_type']
                    unit['knowledge_type'] = new_type
                    self._update_category_index(unit, old_type, new_type)
                
                self.save_knowledge_base()
                print(f"✅ 批量修改完成")
            else:
                print("取消批量修改")
                
        except (ValueError, IndexError):
            print("❌ 无效选择")
    
    
    def _batch_add_concepts(self, indices: List[int]):
        """批量添加概念"""
        new_concept = input("输入要添加的概念: ").strip()
        
        if new_concept:
            for idx in indices:
                unit = self.knowledge_base["knowledge_units"][idx]
                if "key_concepts" not in unit:
                    unit["key_concepts"] = []
                
                if new_concept not in unit["key_concepts"]:
                    unit["key_concepts"].append(new_concept)
            
            self._update_concept_index(self.knowledge_base["knowledge_units"][indices[0]])
            self.save_knowledge_base()
            print(f"✅ 批量添加概念完成")
    
    
    def _batch_adjust_scores(self, indices: List[int]):
        """批量调整评分"""
        print("选择要调整的评分维度:")
        metrics = ["actionability", "specificity", "uniqueness", "clarity"]
        
        for i, m in enumerate(metrics, 1):
            print(f"{i}. {m}")
        
        choice = input("选择维度: ").strip()
        
        try:
            metric = metrics[int(choice) - 1]
            new_score = float(input(f"输入新的评分 (0.0-1.0): ").strip())
            
            if 0.0 <= new_score <= 1.0:
                for idx in indices:
                    unit = self.knowledge_base["knowledge_units"][idx]
                    if "practical_value" not in unit:
                        unit["practical_value"] = {}
                    unit["practical_value"][metric] = new_score
                
                self.save_knowledge_base()
                print(f"✅ 批量调整评分完成")
            else:
                print("❌ 评分必须在0.0-1.0之间")
                
        except (ValueError, IndexError):
            print("❌ 无效输入")
    
    
    def search_knowledge_units(self, keyword: str, category: str = None):
        """搜索知识单元"""
        results = []
        
        for i, unit in enumerate(self.knowledge_base["knowledge_units"]):
            # 检查分类过滤
            if category:
                category_key = self._get_category_key(category)
                if unit["knowledge_type"] != category_key:
                    continue
            
            # 搜索关键词
            search_text = f"{unit.get('title', '')} {unit['content']} {' '.join(unit.get('key_concepts', []))}"
            
            if keyword.lower() in search_text.lower():
                results.append((i, unit))
        
        print(f"\n=== 搜索结果: '{keyword}' ({len(results)}个结果) ===")
        
        for i, (idx, unit) in enumerate(results[:20], 1):  # 显示前20个结果
            print(f"\n[{i}] 索引: {idx}")
            print(f"    来源: {unit['source_document']} (第{unit['page_number']}页)")
            print(f"    类型: {unit['knowledge_type']}")
            print(f"    标题: {unit.get('title', '无标题')[:50]}...")
            print(f"    匹配内容: {self._highlight_keyword(unit['content'][:200], keyword)}...")
        
        if len(results) > 20:
            print(f"\n... 还有 {len(results) - 20} 个结果")
        
        return results
    
    
    def _highlight_keyword(self, text: str, keyword: str) -> str:
        """高亮关键词"""
        import re
        return re.sub(f"({re.escape(keyword)})", r"**1**", text, flags=re.IGNORECASE)
    
    
    def export_to_csv(self, output_path: str = None):
        """导出为CSV格式，便于Excel编辑"""
        if output_path is None:
            output_path = self.knowledge_base_path.replace('.json', '_editable.csv')
        
        # 转换为DataFrame
        df_data = []
        
        for unit in self.knowledge_base["knowledge_units"]:
            row = {
                "index": self.knowledge_base["knowledge_units"].index(unit),
                "source_document": unit["source_document"],
                "page_number": unit["page_number"],
                "title": unit.get("title", ""),
                "knowledge_type": unit["knowledge_type"],
                "content": unit["content"],
                "key_concepts": " | ".join(unit.get("key_concepts", [])),
                "actionability": unit.get("practical_value", {}).get("actionability", 0),
                "specificity": unit.get("practical_value", {}).get("specificity", 0),
                "uniqueness": unit.get("practical_value", {}).get("uniqueness", 0),
                "clarity": unit.get("practical_value", {}).get("clarity", 0),
                "extraction_timestamp": unit["extraction_timestamp"]
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ 知识库已导出到: {output_path}")
        print("您可以在Excel中编辑此文件，然后使用 import_from_csv() 导入")
        
        return output_path
    
    
    def import_from_csv(self, csv_path: str):
        """从CSV导入编辑后的知识库"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 验证必需列
            required_columns = ["index", "knowledge_type", "key_concepts", "actionability", "specificity", "uniqueness", "clarity"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ CSV文件缺少必需列: {missing_columns}")
                return
            
            # 创建备份
            self.create_backup()
            
            # 更新知识单元
            updated_count = 0
            
            for _, row in df.iterrows():
                idx = int(row["index"])
                
                if 0 <= idx < len(self.knowledge_base["knowledge_units"]):
                    unit = self.knowledge_base["knowledge_units"][idx]
                    
                    # 更新基本信息
                    unit["knowledge_type"] = row["knowledge_type"]
                    unit["title"] = str(row.get("title", ""))
                    
                    # 更新关键概念
                    concepts_str = str(row.get("key_concepts", ""))
                    if concepts_str and concepts_str != "nan":
                        unit["key_concepts"] = [c.strip() for c in concepts_str.split("|") if c.strip()]
                    
                    # 更新实用价值
                    if "practical_value" not in unit:
                        unit["practical_value"] = {}
                    
                    unit["practical_value"]["actionability"] = float(row.get("actionability", 0))
                    unit["practical_value"]["specificity"] = float(row.get("specificity", 0))
                    unit["practical_value"]["uniqueness"] = float(row.get("uniqueness", 0))
                    unit["practical_value"]["clarity"] = float(row.get("clarity", 0))
                    
                    updated_count += 1
            
            # 重建索引
            self._rebuild_indices()
            
            # 保存
            self.save_knowledge_base(backup=False)
            
            print(f"✅ 成功导入 {updated_count} 个知识单元的修改")
            
        except Exception as e:
            print(f"❌ 导入失败: {e}")
    
    
    def _rebuild_indices(self):
        """重建所有索引"""
        # 重建分类索引
        category_index = {}
        for i, unit in enumerate(self.knowledge_base["knowledge_units"]):
            category = unit["knowledge_type"]
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(i)
        
        self.knowledge_base["category_index"] = category_index
        
        # 重建概念索引
        concept_index = {}
        for i, unit in enumerate(self.knowledge_base["knowledge_units"]):
            concepts = unit.get("key_concepts", [])
            for concept in concepts:
                if concept not in concept_index:
                    concept_index[concept] = []
                concept_index[concept].append(i)
        
        self.knowledge_base["concept_index"] = concept_index
    
    
    def interactive_editor(self):
        """交互式编辑器主界面"""
        print("\n=== 专家知识编辑器 ===")
        print("欢迎使用专家知识编辑工具！")
        
        while True:
            print("\n主菜单:")
            print("1. 浏览知识单元")
            print("2. 搜索知识单元")
            print("3. 编辑单个知识单元")
            print("4. 批量编辑分类")
            print("5. 导出到CSV")
            print("6. 从CSV导入")
            print("7. 保存并退出")
            print("8. 直接退出")
            
            choice = input("请选择操作 (1-8): ").strip()
            
            if choice == "1":
                category = input("输入分类名称（留空显示全部）: ").strip()
                limit = int(input("显示数量 (默认10): ") or "10")
                offset = int(input("起始位置 (默认0): ") or "0")
                self.list_knowledge_units(category=category if category else None, limit=limit, offset=offset)
                
            elif choice == "2":
                keyword = input("输入搜索关键词: ").strip()
                category = input("限制分类（留空搜索全部）: ").strip()
                self.search_knowledge_units(keyword, category=category if category else None)
                
            elif choice == "3":
                unit_index = int(input("输入知识单元索引: "))
                self.edit_knowledge_unit(unit_index)
                
            elif choice == "4":
                category = input("输入要批量编辑的分类: ").strip()
                self.batch_edit_by_category(category)
                
            elif choice == "5":
                output_path = input("输出文件路径（留空使用默认）: ").strip()
                self.export_to_csv(output_path if output_path else None)
                
            elif choice == "6":
                csv_path = input("CSV文件路径: ").strip()
                if csv_path:
                    self.import_from_csv(csv_path)
                
            elif choice == "7":
                self.save_knowledge_base()
                print("感谢使用专家知识编辑器！")
                break
                
            elif choice == "8":
                confirm = input("确定要退出吗？未保存的修改将丢失 (y/n): ").strip().lower()
                if confirm == 'y':
                    print("退出编辑器")
                    break
                    
            else:
                print("❌ 无效选择")


def main():
    """主函数"""
    print("=== 专家知识编辑系统 ===")
    
    # 创建编辑器
    editor = ExpertKnowledgeEditor()
    
    if editor.knowledge_base is None:
        print("无法加载知识库，程序退出")
        return
    
    # 启动交互式编辑器
    editor.interactive_editor()


if __name__ == "__main__":
    main()