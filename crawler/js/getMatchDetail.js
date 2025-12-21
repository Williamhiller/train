/**
 * 抓取比赛详情，包括历史交锋和赔率信息
 * 输入参数：matchId
 * 自动生成历史交锋和赔率信息的URL
 */

var getAllMatch = require("./getAllMatch");
var fs = require('fs');
var path = require('path');

// 公司名称映射
const company = {
    82: 'Ladbrokes',
    115: 'william'
};

// 格式化函数：将数值保留两位小数
const formatDecimal = (value) => {
    return Number(value).toFixed(2);
};

// 数据解析函数：从原始数据数组中提取所需字段
const parseMatchData = (matchArray) => {
    return [
        matchArray[0],     // 比赛时间
        matchArray[1],     // 联赛id
        matchArray[4],     // 主队id
        matchArray[6],     // 客队id
        matchArray[8],     // 主队进球数
        matchArray[9],     // 客队进球数
        matchArray[12]     // 赛果
    ];
};

// 解析赔率数据并进行结构调整
// season参数：赛季信息，如"2023-2024"
const parseOddsData = (detail, season) => {
    const detailParts = detail.split('^');
    if (detailParts.length < 2) {
        console.log('赔率详情格式错误');
        return [];
    }

    const oddsData = detailParts[1];
    const oddsArray = oddsData.split(';').filter(item => item.trim() !== '');

    // 解析赛季信息，获取开始年份和结束年份
    const seasonYears = season.split('-').map(year => parseInt(year, 10));
    const startYear = seasonYears[0];
    const endYear = seasonYears[1];

    return oddsArray.map(odds => {
        const parts = odds.split('|');
        
        // 解析时间部分，获取月份
        const timePart = parts[3];
        const monthMatch = timePart.match(/(\d{2})-(\d{2})/);
        
        let year;
        if (monthMatch) {
            const month = parseInt(monthMatch[1], 10);
            // 根据月份确定年份：大于6月使用开始年份，否则使用结束年份
            year = (month > 5) ? startYear : endYear;
        } else {
            // 如果无法解析月份，默认使用结束年份
            year = endYear;
        }
        
        const fullTime = `${year}-${timePart}`;

        // 当parts长度不足7时，胜平负赔付比例默认使用0
        return [
            formatDecimal(parts[0]),  // 胜赔率（保留两位小数）
            formatDecimal(parts[1]),  // 平赔率（保留两位小数）
            formatDecimal(parts[2]),  // 负赔率（保留两位小数）
            formatDecimal(parts[4] || 0),  // 胜赔付比例（保留两位小数）
            formatDecimal(parts[5] || 0),  // 平赔付比例（保留两位小数）
            formatDecimal(parts[6] || 0),  // 负赔付比例（保留两位小数）
            fullTime   // 完整时间（包含年份）
        ];
    }).filter(item => item.length > 0);
}; 

// 处理数组中的每个子项，确保子项内部没有换行
const processDataArray = (dataArray) => {
    const dataStrings = dataArray.map(item => JSON.stringify(item).replace(/\n/g, '').replace(/\s+/g, ' '));
    return '\n    ' + dataStrings.join(',\n    ') + '\n  ';
};

// 主函数：获取比赛详情
// savePath 可选参数，指定保存路径，如果不提供则不保存到文件
// season 可选参数，指定赛季信息，如果不提供则使用默认值2023-2024
const getMatchDetail = async (matchId, savePath = null, season = null) => {
    // console.log(`开始获取比赛详情，matchId: ${matchId}`);
    
    // 根据matchId生成对应的URL
    const historyUrl = `https://zq.titan007.com/analysis/${matchId}cn.htm`;
    const oddUrl = `https://1x2.titan007.com/oddslist/${matchId}.htm`;
    
    // console.log(`历史交锋URL: ${historyUrl}`);
    // console.log(`赔率信息URL: ${oddUrl}`);
    
    // 同时获取历史交锋数据和赔率数据
    console.log('开始获取历史交锋数据...');
    const historyData = await getAllMatch(historyUrl, 'analyze');
    
    console.log('开始获取赔率数据...');
    const oddData = await getAllMatch(oddUrl, 'odd');
    
    // 检查数据是否获取成功
    if (!historyData || !oddData) {
        console.error('未能获取到完整的比赛数据');
        return null;
    }
    
    // 处理历史交锋数据
    const {awayData=[], homeData=[], historyData: rawHistoryData=[]} = historyData;
    
    // 解析历史交锋数据
    const parsedAwayData = awayData.map(parseMatchData);
    const parsedHomeData = homeData.map(parseMatchData);
    const parsedHistoryData = rawHistoryData.map(parseMatchData);
    
    // 处理赔率数据
    const {game=[], gameDetail=[]} = oddData;
    
    // 遍历game数组，找到82和115的项
    const targetOddIds = {};
    for (let item of game) {
        const parts = item.split('|');
        if (parts.length >= 2 && (parts[0] === '82' || parts[0] === '115')) {
            targetOddIds[parts[0]] = parts[1];
        }
    }
    
    // 检查是否找到82和115的项
    if (!targetOddIds['82'] || !targetOddIds['115']) {
        console.log('未找到目标赔率ID');
        return null;
    }
    
    // 遍历gameDetail数组，找到对应的项
    const targetDetails = {};
    for (let item of gameDetail) {
        for (let key in targetOddIds) {
            if (item.startsWith(targetOddIds[key] + '^')) {
                targetDetails[key] = item;
                break;
            }
        }
    }
    
    // 检查是否找到对应的项
    if (!targetDetails['82'] || !targetDetails['115']) {
        console.log('未找到目标赔率详情');
        return null;
    }
    
    // 解析82和115的赔率数据，使用传入的赛季信息或默认值
    const currentSeason = season || '2023-2024';
    const parsedOdds82 = parseOddsData(targetDetails['82'], currentSeason);
    const parsedOdds115 = parseOddsData(targetDetails['115'], currentSeason);
    
    // 构建完整的结果对象
    const result = {
        matchId: matchId,
        history: {
            awayData: parsedAwayData,
            homeData: parsedHomeData,
            historyData: parsedHistoryData
        },
        odds: {
            oddId: {
                '82': targetOddIds['82'],
                '115': targetOddIds['115']
            },
            '82': parsedOdds82,
            '115': parsedOdds115
        }
    };
    
    // 构建完整的JSON字符串，保持与getMatchOdd.js相同的风格
    const awayDataContent = processDataArray(result.history.awayData);
    const homeDataContent = processDataArray(result.history.homeData);
    const historyDataContent = processDataArray(result.history.historyData);
    const odds82Content = processDataArray(result.odds['82']);
    const odds115Content = processDataArray(result.odds['115']);

    const jsonStr = `{
  "matchId": "${result.matchId}",
  "history": {
    "awayData": [${awayDataContent}],
    "homeData": [${homeDataContent}],
    "historyData": [${historyDataContent}]
  },
  "odds": {
    "oddId": {
      "82": "${result.odds.oddId['82']}",
      "115": "${result.odds.oddId['115']}"
    },
    "82": [${odds82Content}],
    "115": [${odds115Content}]
  }
}`;
    
    // 如果提供了保存路径，则保存到文件
    if (savePath) {
        fs.writeFileSync(savePath, jsonStr);
        console.log('比赛详情数据已保存到:', savePath);
    }
    
    return result;
};

// 如果直接运行此文件，则使用命令行参数传入的matchId，否则使用默认值
if (require.main === module) {
    // 从命令行参数获取matchId，如果没有提供则使用默认值
    const matchId = process.argv[2] || '2590911';
    
    console.log(`正在测试matchId: ${matchId}`);
    
    getMatchDetail(matchId).catch((error) => {
        console.error('执行出错:', error);
    });
}

module.exports = getMatchDetail;
