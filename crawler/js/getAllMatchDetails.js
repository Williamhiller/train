/**
 * 读取指定赛季目录下的文件，获取所有比赛信息
 * 遍历获取每场比赛的详情，并保存为round-matchId名称的文件
 * 添加延时机制以避免被识别为爬虫
 */

var getMatchDetail = require("./getMatchDetail");
var fs = require('fs');
var path = require('path');

/**
 * 延时函数，随机生成延时时间，避免被识别为爬虫
 * @param min 最小延时时间（毫秒）
 * @param max 最大延时时间（毫秒）
 * @returns {Promise<void>}
 */
function delay(min, max) {
    const delayTime = Math.floor(Math.random() * (max - min + 1)) + min;
    console.log(`等待 ${delayTime} 毫秒后继续...`);
    return new Promise(resolve => setTimeout(resolve, delayTime));
}

/**
 * 读取指定赛季目录下的文件，获取所有比赛信息
 * @param season 赛季名称，如"2024-2025"
 * @returns {Promise<Object>}
 */
function readSeasonData(season) {
    return new Promise((resolve, reject) => {
        const seasonDir = path.join(__dirname, 'output', season);
        
        // 检查赛季目录是否存在
        if (!fs.existsSync(seasonDir)) {
            reject(new Error(`赛季目录 ${seasonDir} 不存在`));
            return;
        }
        
        // 读取赛季目录下的所有文件
        fs.readdir(seasonDir, (err, files) => {
            if (err) {
                reject(err);
                return;
            }
            
            // 过滤出JSON文件
            const jsonFiles = files.filter(file => file.endsWith('.json'));
            
            if (jsonFiles.length === 0) {
                reject(new Error(`赛季目录 ${seasonDir} 下没有JSON文件`));
                return;
            }
            
            // 读取第一个JSON文件（假设每个赛季只有一个JSON文件）
            const dataFile = jsonFiles[0];
            const dataFilePath = path.join(seasonDir, dataFile);
            
            console.log(`正在读取赛季数据文件: ${dataFilePath}`);
            
            fs.readFile(dataFilePath, 'utf8', (err, data) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                try {
                    const matchData = JSON.parse(data);
                    resolve(matchData);
                } catch (parseError) {
                    reject(new Error(`解析JSON文件 ${dataFilePath} 失败: ${parseError.message}`));
                }
            });
        });
    });
}

/**
 * 为指定比赛创建详细信息目录
 * @param season 赛季名称
 * @param round 轮次
 * @param matchId 比赛ID
 * @returns {string} 创建的目录路径
 */
function createMatchDetailDir(season, round, matchId) {
    // 创建赛季详细信息目录
    const seasonDetailDir = path.join(__dirname, 'output', season, 'details');
    if (!fs.existsSync(seasonDetailDir)) {
        fs.mkdirSync(seasonDetailDir);
    }
    
    // 创建轮次目录
    const roundDir = path.join(seasonDetailDir, round);
    if (!fs.existsSync(roundDir)) {
        fs.mkdirSync(roundDir);
    }
    
    return roundDir;
}

/**
 * 主函数，读取赛季数据并遍历获取每场比赛的详情
 */
async function main() {
    const season = '2015-2016'; // 要处理的赛季
    let processedMatches = 0;
    let failedMatches = 0;
    let failedMatchesList = [];
    
    try {
        console.log(`开始处理 ${season} 赛季的所有比赛详情...`);
        
        // 读取赛季数据
        const matchData = await readSeasonData(season);
        const totalMatches = Object.keys(matchData).length;
        
        console.log(`成功读取赛季数据，共 ${totalMatches} 场比赛`);
        
        // 遍历所有比赛，获取详细信息
        let index = 0;
        for (const matchId in matchData) {
            if (matchData.hasOwnProperty(matchId)) {
                const matchInfo = matchData[matchId];
                const round = matchInfo.round;
                
                index++;
                console.log(`\n正在处理第 ${index}/${totalMatches} 场比赛`);
                console.log(`比赛ID: ${matchId}, 轮次: ${round}`);
                console.log(`主队: ${matchInfo.homeTeamName}, 客队: ${matchInfo.awayTeamName}`);
                
                try {
                    // 创建比赛详细信息目录
                    const roundDir = createMatchDetailDir(season, round, matchId);
                    
                    // 生成文件名：round-matchId.json
                    const fileName = `${matchId}.json`;
                    const filePath = path.join(roundDir, fileName);
                    
                    // 检查文件是否已存在，如果存在则跳过
                    if (fs.existsSync(filePath)) {
                        console.log(`文件已存在，跳过抓取: ${filePath}`);
                        processedMatches++;
                    } else {
                        // 获取比赛详情并保存到指定路径，传入赛季信息
                            const detail = await getMatchDetail(matchId, filePath, season);
                        
                        if (detail) {
                            processedMatches++;
                        } else {
                            console.log(`获取比赛详情失败`);
                            failedMatches++;
                            failedMatchesList.push({ matchId, round });
                        }

                        // 如果不是最后一场比赛，添加延时
                        if (index < totalMatches) {
                            // 添加3-5秒的随机延时，避免被识别为爬虫
                            await delay(500, 2000);
                        }
                    }
                    

                    
                } catch (error) {
                    console.error(`处理比赛 ${matchId} 时出错:`, error.message);
                    failedMatches++;
                    failedMatchesList.push({ matchId, round });
                    
                    // 即使出错也继续处理下一场比赛，并添加延时
                    // if (index < totalMatches) {
                    //     await delay(1000, 2000); // 出错后使用较短的延时
                    // }
                }
            }
        }
        
        console.log(`\n=== 处理完成 ===`);
        console.log(`总比赛数: ${totalMatches}`);
        console.log(`成功处理: ${processedMatches}`);
        console.log(`处理失败: ${failedMatches}`);
        if (failedMatchesList.length > 0) {
            console.log(`\n失败的比赛记录:`);
            failedMatchesList.forEach((failedMatch, index) => {
                console.log(`${index + 1}. 轮次: ${failedMatch.round}, 比赛ID: ${failedMatch.matchId}`);
            });
            console.log(`\n失败记录总数: ${failedMatchesList.length}`);
        }
        console.log(`================`);
        
    } catch (error) {
        console.error(`执行过程中发生错误:`, error.message);
    }
}

// 启动主函数
main().catch(error => {
    console.error('执行过程中发生错误:', error);
});
