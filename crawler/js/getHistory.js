
var getAllMatch = require("./getAllMatch");
var fs = require('fs');
var path = require('path');

// 从URL中提取matchId
const url = "https://zq.titan007.com/analysis/2590911cn.htm"; 
const matchIdMatch = url.match(/\d+(?=cn\.htm$)/);
const matchId = matchIdMatch ? matchIdMatch[0] : null;

// 检查matchId是否成功提取
if (!matchId) {
    console.error('无法从URL中提取比赛ID');
    process.exit(1);
}

const getHistory = async () => {
    console.log('开始获取比赛数据...');
    const pageData = await getAllMatch(url, 'analyze');
    console.log('获取到的原始数据:', pageData);
    
    if (!pageData) {
        console.error('未能获取到比赛数据');
        return null;
    }
    const {awayData=[],homeData=[],historyData=[]} = pageData;
    // 数据格式为20位长度的数组，分别是 比赛时间、联赛id、联赛名称、主队颜色、主队id、主队名称、客队id、客队名称、主队进球数、客队进球数、半场比分、状态、主队进球差、客队进球差、主队进球数差、赛果、比赛id、主队角球数、客队角球数、比赛链接、状态
    // ["24-08-04",41,"球会友谊","#00A8A8",35,"<span title="排名：英超10">水晶宫(中)</span>",62,"<span title="排名：英超9">西汉姆联</span>",3,1,"1-1","",1,-2,1,2640310,"7","4","//zq.titan007.com/cn/cupmatch.aspx?sclassid=41",0]

    // 数据解析函数：从原始数据数组中提取所需字段
    const parseMatchData = (matchArray) => {
        // 字段索引：0-比赛时间，1-联赛id，4-主队id，6-客队id，8-主队进球数，9-客队进球数，17-赛果
        // 返回数组形式，顺序：比赛时间、联赛id、主队id、客队id、主队进球数、客队进球数、赛果
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

    // 解析所有数据
    const parsedData = {
        awayData: awayData.map(parseMatchData),
        homeData: homeData.map(parseMatchData),
        historyData: historyData.map(parseMatchData)
    };

    // 处理数组中的每个子项，确保子项内部没有换行，格式更紧凑
    const processDataArray = (dataArray) => {
        const dataStrings = dataArray.map(item => JSON.stringify(item).replace(/\n/g, '').replace(/\s+/g, ' '));
        return '\n    ' + dataStrings.join(',\n    ') + '\n  ';
    };

    // 构建完整的JSON字符串，参考getMatchOdd.js的格式
    const awayDataContent = processDataArray(parsedData.awayData);
    const homeDataContent = processDataArray(parsedData.homeData);
    const historyDataContent = processDataArray(parsedData.historyData);

    const jsonStr = `{
  "awayData": [${awayDataContent}],
  "homeData": [${homeDataContent}],
  "historyData": [${historyDataContent}]
}`;

    // 保存到JSON文件
    const outputPath = path.join(__dirname, 'output', 'historyData.json');
    fs.writeFileSync(outputPath, jsonStr);
    console.log('数据已保存到:', outputPath);

    return parsedData;
}

// 调用getHistory函数执行数据抓取和保存
if (require.main === module) {
    getHistory().catch((error) => {
        console.error('执行出错:', error);
    });
}