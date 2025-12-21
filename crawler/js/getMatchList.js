/**
 * Created by Williamhiler on 2016/11/22.
 * Updated to save data to local JSON file
 * Modified to support season traversal from 2016 to 2024-2025 with delay mechanism
 */

var getAllMatch = require("./getAllMatch");
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
 * 处理单个赛季的数据
 * @param season 赛季名称
 * @param leagueId 联赛ID
 * @returns {Promise<void>}
 */
async function processSeason(season, leagueId) {
    try {
        console.log(`开始处理 ${season} 赛季的数据...`);
        const url = `http://zq.titan007.com/cn/League/${season}/${leagueId}.html`;
        console.log(`正在访问URL: ${url}`);
        const pageData = await getAllMatch(url);
        
        // 检查pageData是否有效
        if (!pageData || typeof pageData !== 'object') {
            console.error(`${season} 赛季返回的数据无效`);
            return 0;
        }
        
        // 构建球队字典，处理arrTeam可能不存在的情况
        var teamDic = {};
        if (Array.isArray(pageData.arrTeam)) {
            pageData.arrTeam.forEach(function (item) {
                if (item && Array.isArray(item) && item.length >= 2) {
                    teamDic[item[0]] = item[1];
                }
            });
        }
        
        // 创建key-value格式的数据结构，只保存比赛代码、联赛代码、主队名称、客队名称
        var matchData = {};
        
        // 循环处理每个比赛，处理jh可能不存在的情况
        if (pageData.jh && typeof pageData.jh === 'object') {
            Object.keys(pageData.jh).forEach(function (key) {
                var round = pageData.jh[key];
                if (Array.isArray(round)) {
                    round.forEach(function (item) {
                        if (Array.isArray(item) && item.length >= 6) {
                            // 获取比赛代码作为key
                            var matchId = item[0];
                            
                            // 检查是否有比分，没有比分则跳过
                            const score = item[6];
                            if (!score || score === '' || score === '-') {
                                console.log(`跳过无比分的比赛: ${matchId}`);
                                return;
                            }
                            
                            // 解析比分并判断赛果
                            const scoreParts = score.split('-');
                            if (scoreParts.length === 2) {
                                const homeScore = parseInt(scoreParts[0], 10);
                                const awayScore = parseInt(scoreParts[1], 10);
                                
                                let matchResult = '';
                                if (homeScore > awayScore) {
                                    matchResult = 3; // 主队胜
                                } else if (homeScore < awayScore) {
                                    matchResult = 0; // 客队胜
                                } else {
                                    matchResult = 1; // 平局
                                }
                                
                                // 创建包含所需字段的对象作为value
                                matchData[matchId] = {
                                    round: key,
                                    matchId: item[0],  
                                    season: season,             // 赛季信息
                                    homeTeamName: teamDic[item[4]] || '未知',  // 主队名称，从teamDic获取
                                    awayTeamName: teamDic[item[5]] || '未知',   // 客队名称，从teamDic获取
                                    homeTeamId: item[4],
                                    awayTeamId: item[5],
                                    score: score, // 比分
                                    result: matchResult, // 根据比分判断的赛果
                                    homeScore: homeScore, // 主队进球数
                                    awayScore: awayScore, // 客队进球数
                                };
                            } else {
                                console.log(`跳过比分格式无效的比赛: ${matchId}, 比分: ${score}`);
                                return;
                            }
                        }
                    });
                }
            });
        }
        
        console.log(`${season} 赛季处理完成，共获取比赛数量:`, Object.keys(matchData).length);
        
        // 保存数据到本地JSON文件，使用默认联赛信息如果arrLeague不存在
        const leagueInfo = Array.isArray(pageData.arrLeague) ? pageData.arrLeague : [leagueId, '英格兰超级联赛'];
        saveDataToJson(matchData, leagueInfo, season);
        
        // 返回处理的比赛数量
        return Object.keys(matchData).length;
        
    } catch (error) {
        console.error(`处理 ${season} 赛季时出错:`, error.message);
        // 添加更详细的错误信息
        if (error.stack) {
            console.error('错误堆栈:', error.stack.substring(0, 200)); // 只显示部分堆栈，避免输出过长
        }
        return 0;
    }
}

/**
 * 保存数据到本地JSON文件
 * @param data 要保存的数据对象
 * @param arrLeague 包含联赛ID和名称的数组，用于文件名
 * @param season 赛季信息
 */
function saveDataToJson(data, arrLeague, season) {
    // 确保输出目录存在
    var outputDir = path.join(__dirname, 'output');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }
    
    // 创建以赛季名称命名的子目录
    var seasonDir = path.join(outputDir, season);
    if (!fs.existsSync(seasonDir)) {
        fs.mkdirSync(seasonDir);
    }
    
    // 生成文件名 (移除特殊字符)
    var fileName = `${arrLeague[0]}_${season}.json`;
    var filePath = path.join(seasonDir, fileName);
    
    // 写入文件
    fs.writeFile(filePath, JSON.stringify(data, null, 2), function(err) {
        if (err) {
            console.error('保存文件失败:', err);
        } else {
            console.log('数据已成功保存至:', filePath);
        }
    });
}

/**
 * 主函数，遍历处理多个赛季的数据
 */
async function main() {
    // 定义要处理的赛季列表，从最近的赛季开始处理，更容易测试
    const seasons = [
        // '2024-2025',  // 先处理最近的赛季
        '2023-2024',
        // '2022-2023',
        // '2021-2022',
        // '2020-2021',
        // '2019-2020',
        // '2018-2019',
        // '2017-2018',
        // '2016-2017',
        // '2015-2016',
    ];
    
    const leagueId = 36; // 英格兰超级联赛ID
    let totalMatches = 0;
    let successfulSeasons = 0;
    
    console.log('开始遍历赛季数据...');
    
    // 依次处理每个赛季
    for (let i = 0; i < seasons.length; i++) {
        const season = seasons[i];
        
        try {
            // 处理当前赛季
            const matchCount = await processSeason(season, leagueId);
            
            if (matchCount > 0) {
                successfulSeasons++;
                totalMatches += matchCount;
                console.log(`✅ ${season} 赛季成功处理`);
            } else {
                console.log(`⚠️ ${season} 赛季未获取到数据或处理失败`);
            }
            
            // 如果不是最后一个赛季，添加延时
            if (i < seasons.length - 1) {
                // 添加10-20秒的随机延时（减少测试时间），避免被识别为爬虫
                await delay(10000, 20000);
            }
        } catch (error) {
            console.error(`❌ 处理 ${season} 赛季时发生异常:`, error.message);
            // 即使出错也继续处理下一个赛季
            if (i < seasons.length - 1) {
                await delay(5000, 10000); // 出错后使用较短的延时
            }
        }
    }
    
    console.log(`\n=== 处理完成 ===`);
    console.log(`成功处理的赛季数量: ${successfulSeasons}/${seasons.length}`);
    console.log(`总计获取比赛数量: ${totalMatches}`);
    console.log(`================`);
}

// 启动主函数
main().catch(error => {
    console.error('执行过程中发生错误:', error);
});







