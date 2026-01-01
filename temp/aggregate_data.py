import json
import os
import sys

# 配置路径
examples_dir = f"/Users/Williamhiler/Documents/my-project/train/examples"

# 确保examples目录存在
os.makedirs(examples_dir, exist_ok=True)

# 获取所有联赛和赛季
def get_all_leagues_seasons():
    original_data_root = "/Users/Williamhiler/Documents/my-project/train/original-data"
    leagues = [d for d in os.listdir(original_data_root) if os.path.isdir(os.path.join(original_data_root, d)) and not d.startswith('.')]
    
    leagues_seasons = []
    for league in leagues:
        league_dir = os.path.join(original_data_root, league)
        seasons = [d for d in os.listdir(league_dir) if os.path.isdir(os.path.join(league_dir, d)) and not d.startswith('.')]
        for season in seasons:
            leagues_seasons.append((league, season))
    
    return leagues_seasons

# 读取主数据文件
def read_main_data(league, season):
    original_data_dir = f"/Users/Williamhiler/Documents/my-project/train/original-data/{league}/{season}"
    
    # 自动检测数据文件
    files = [f for f in os.listdir(original_data_dir) if f.endswith(f"_{season}.json")]
    if not files:
        raise FileNotFoundError(f"未找到{league}{season}赛季的数据文件")
    
    main_file = os.path.join(original_data_dir, files[0])
    with open(main_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取比赛详情
def read_match_details(league, season, match_id, round_name):
    original_data_dir = f"/Users/Williamhiler/Documents/my-project/train/original-data/{league}/{season}"
    details_file = os.path.join(original_data_dir, "details", round_name, f"{match_id}.json")
    if os.path.exists(details_file):
        with open(details_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# 转换日期格式：24-08-10 → 2024-08-10
def convert_date(date_str):
    if len(date_str) == 8 and date_str[2] == '-':
        year = date_str[:2]
        month = date_str[3:5]
        day = date_str[6:8]
        return f"20{year}-{month}-{day}"
    return date_str

# 清洗比赛详情数据
def clean_details(details):
    # 移除重复的matchId
    if "matchId" in details:
        del details["matchId"]
    
    # 移除oddId字段
    if "odds" in details and "oddId" in details["odds"]:
        del details["odds"]["oddId"]
    
    # 转换历史数据中的日期格式
    if "history" in details:
        history = details["history"]
        
        # 处理awayData
        if "awayData" in history:
            for i, record in enumerate(history["awayData"]):
                if isinstance(record[0], str):
                    history["awayData"][i][0] = convert_date(record[0])
        
        # 处理homeData
        if "homeData" in history:
            for i, record in enumerate(history["homeData"]):
                if isinstance(record[0], str):
                    history["homeData"][i][0] = convert_date(record[0])
        
        # 处理historyData
        if "historyData" in history:
            for i, record in enumerate(history["historyData"]):
                if isinstance(record[0], str):
                    history["historyData"][i][0] = convert_date(record[0])
    
    return details

# 聚合数据
def aggregate_data(league, season):
    main_data = read_main_data(league, season)
    aggregated_data = {}
    
    for match_key, match_info in main_data.items():
        match_id = match_info["matchId"]
        round_name = match_info["round"]
        
        # 读取比赛详情
        details = read_match_details(league, season, match_id, round_name)
        
        # 聚合数据
        if details:
            # 清洗详情数据
            cleaned_details = clean_details(details)
            aggregated_data[match_key] = {
                **match_info,
                "details": cleaned_details
            }
        else:
            aggregated_data[match_key] = match_info
    
    return aggregated_data

# 压缩数据：移除不必要的字段，优化数据结构
def compress_data(data):
    compressed = {}
    for match_key, match_info in data.items():
        # 基础字段保留，只保留训练必需的字段
        compressed_match = {
            "matchId": match_info["matchId"],
            "round": match_info["round"],
            "season": match_info["season"],
            "matchTime": match_info["matchTime"],
            "homeTeamId": match_info["homeTeamId"],
            "awayTeamId": match_info["awayTeamId"],
            "result": match_info["result"],
            "homeScore": match_info["homeScore"],
            "awayScore": match_info["awayScore"]
        }
        
        # 只保留关键的历史数据，移除冗余信息
        if "details" in match_info:
            details = match_info["details"]
            compressed_details = {}
            
            # 处理history数据：只保留最近5场比赛，移除完整历史
            if "history" in details:
                history = details["history"]
                compressed_history = {}
                
                # 只保留最近5场比赛数据
                for key in ["homeData", "awayData", "historyData"]:
                    if key in history:
                        # 只保留最近5场比赛
                        compressed_history[key] = history[key][:6] if len(history[key]) > 6 else history[key]
                
                # 保留赛季数据
                if "homeSeasonData" in history:
                    compressed_history["homeSeasonData"] = history["homeSeasonData"]
                if "awaySeasonData" in history:
                    compressed_history["awaySeasonData"] = history["awaySeasonData"]
                
                if compressed_history:
                    compressed_details["history"] = compressed_history
            
            # 保留完整的赔率数据，包括所有赔率变化
            if "odds" in details:
                odds = details["odds"]
                compressed_odds = {}
                
                # 保留所有赔率变化记录
                for bookmaker_id, odds_data in odds.items():
                    if isinstance(odds_data, list) and odds_data:
                        # 保留完整的赔率历史记录
                        compressed_odds[bookmaker_id] = odds_data
                
                if compressed_odds:
                    compressed_details["odds"] = compressed_odds
            
            if compressed_details:
                compressed_match["details"] = compressed_details
        
        compressed[match_key] = compressed_match
    
    return compressed

# 保存聚合后的数据
def save_aggregated_data(data, league, season):
    # 压缩数据结构
    compressed_data = compress_data(data)
    
    # 只保存紧凑JSON格式（去除空格和缩进）
    json_output = os.path.join(examples_dir, f"{league}_{season}_aggregated.json")
    with open(json_output, 'w', encoding='utf-8') as f:
        # 使用separators=(',', ':')去除空格，生成紧凑JSON
        json.dump(compressed_data, f, separators=(',', ':'), ensure_ascii=False)
    
    # 计算文件大小对比
    original_file = f"/Users/Williamhiler/Documents/my-project/train/original-data/{league}/{season}/36_{season}.json"
    if os.path.exists(original_file):
        original_size = os.path.getsize(original_file)
    else:
        original_file = f"/Users/Williamhiler/Documents/my-project/train/original-data/{league}/{season}/37_{season}.json"
        original_size = os.path.getsize(original_file) if os.path.exists(original_file) else 0
    
    json_size = os.path.getsize(json_output)
    
    print(f"{league}{season}赛季数据处理完成:")
    print(f"  原始数据: {original_size / 1024:.2f} KB")
    print(f"  紧凑JSON: {json_size / 1024:.2f} KB")
    print(f"  数据精简率: {(1 - json_size / original_size) * 100:.2f}%" if original_size > 0 else "  无法计算精简率")

if __name__ == "__main__":
    print("开始批量处理所有联赛和赛季的数据...")
    
    # 获取所有联赛和赛季
    leagues_seasons = get_all_leagues_seasons()
    total = len(leagues_seasons)
    
    for i, (league, season) in enumerate(leagues_seasons, 1):
        print(f"\n处理进度: {i}/{total} - {league}{season}赛季")
        print(f"开始聚合{league}{season}赛季数据...")
        try:
            aggregated_data = aggregate_data(league, season)
            save_aggregated_data(aggregated_data, league, season)
        except Exception as e:
            print(f"处理{league}{season}赛季数据时出错: {e}")
    
    print("\n所有数据处理完成！")
    print("已生成紧凑格式的JSON文件（去除空格），便于训练使用。")
