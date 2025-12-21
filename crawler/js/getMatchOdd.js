
var getAllMatch = require("./getAllMatch");
var fs = require('fs');
var path = require('path');

// 从URL中提取matchId
const url = "https://1x2.titan007.com/oddslist/2590911.htm"; 
const matchId = url.match(/\d+(?=\.htm$)/)[0];
const company = {
    82: 'Ladbrokes',
    115: 'william'
};
const getMatchOdd = async () => {
    const pageData = await getAllMatch(url, 'odd');
    if (!pageData) {
        console.log(pageData);
        return null;
    }

    const {game=[],gameDetail=[]} = pageData;
    
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

    // 格式化函数：将数值保留两位小数
    const formatDecimal = (value) => {
        // 先将字符串转换为浮点数，然后保留两位小数
        return Number(value).toFixed(2);
    };

    // 解析赔率数据并进行结构调整
    const parseOddsData = (detail) => {
        const detailParts = detail.split('^');
        if (detailParts.length < 2) {
            console.log('赔率详情格式错误');
            return [];
        }

        const oddsData = detailParts[1];
        const oddsArray = oddsData.split(';').filter(item => item.trim() !== '');

        return oddsArray.map(odds => {
            const parts = odds.split('|');
            if (parts.length >= 8) {
                // 前三个值：胜、平、负的赔率
                // 接下来三个值：胜、平、负的赔付比例
                // 拼接时间和年份
                const fullTime = `${parts[7]}-${parts[3]}`;

                return [
                    formatDecimal(parts[0]),  // 胜赔率（保留两位小数）
                    formatDecimal(parts[1]),  // 平赔率（保留两位小数）
                    formatDecimal(parts[2]),  // 负赔率（保留两位小数）
                    formatDecimal(parts[4]),  // 胜赔付比例（保留两位小数）
                    formatDecimal(parts[5]),  // 平赔付比例（保留两位小数）
                    formatDecimal(parts[6]),  // 负赔付比例（保留两位小数）
                    fullTime   // 完整时间（包含年份）
                ];
            }
            return [];
        }).filter(item => item.length > 0);
    };

    // 解析82和115的赔率数据
    const parsedOdds82 = parseOddsData(targetDetails['82']);
    const parsedOdds115 = parseOddsData(targetDetails['115']);

    // 构建结果对象，以82和115作为key，并增加oddId字段
    const result = {
        matchId: matchId,
        oddId: {
            '82': targetOddIds['82'],
            '115': targetOddIds['115']
        },
        '82': parsedOdds82,
        '115': parsedOdds115
    };

    // 将结果存储到本地JSON文件
    // 先正常序列化，然后处理parsedOdds数组的格式，使子项内容不换行
    // 处理parsedOdds数组中的每个子项，确保子项内部没有换行
    const processOddsArray = (oddsArray) => {
        const oddsStrings = oddsArray.map(item => JSON.stringify(item).replace(/\n/g, '').replace(/\s+/g, ' '));
        return '\n    ' + oddsStrings.join(',\n    ') + '\n  ';
    };

    // 构建完整的JSON字符串
    const parsedOddsContent82 = processOddsArray(result['82']);
    const parsedOddsContent115 = processOddsArray(result['115']);

    const jsonStr = `{
  "matchId": "${result.matchId}",
  "oddId": {
    "82": "${result.oddId['82']}",
    "115": "${result.oddId['115']}"
  },
  "82": [${parsedOddsContent82}],
  "115": [${parsedOddsContent115}]
}`;

    // 输出的文件名以matchId命名
    const outputPath = path.join(__dirname, 'output', `${matchId}.json`);
    fs.writeFileSync(outputPath, jsonStr);

    console.log('数据解析和存储完成');
    console.log('结果存储在:', outputPath);
}

getMatchOdd();