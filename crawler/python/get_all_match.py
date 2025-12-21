#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Williamhiler on 2024-12-13
核心爬虫功能，用于获取网页数据
"""

import asyncio
from pyppeteer import launch
import json

async def get_all_match(url, type=''):
    """
    核心爬虫函数，使用pyppeteer访问网页并提取数据
    
    :param url: 要访问的URL
    :param type: 数据类型，'odd'或'analyze'
    :return: 提取的数据
    """
    try:
        # 启动浏览器
        browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.newPage()
        
        # 设置User-Agent
        await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.159 Safari/537.36')
        
        # 访问URL
        await page.goto(url, {'waitUntil': 'networkidle0'})
        
        # 执行JavaScript代码获取数据
        if type == 'odd':
            # 获取赔率数据
            data = await page.evaluate('''() => {
                const data = {};
                
                // 转换对象函数
                const convertObject = function(obj) {
                    if (typeof obj !== 'undefined' && obj !== null) {
                        try {
                            return JSON.parse(JSON.stringify(obj));
                        } catch (e) {
                            console.log('转换对象时出错:', e);
                            return [];
                        }
                    }
                    return [];
                };
                
                // 详情数据
                data.game = convertObject(window.game);
                data.gameDetail = convertObject(window.gameDetail);
                return data;
            }''')
        elif type == 'analyze':
            # 获取分析数据
            data = await page.evaluate('''() => {
                const data = {};
                
                // 转换对象函数
                const convertObject = function(obj) {
                    if (typeof obj !== 'undefined' && obj !== null) {
                        try {
                            return JSON.parse(JSON.stringify(obj));
                        } catch (e) {
                            console.log('转换对象时出错:', e);
                            return [];
                        }
                    }
                    return [];
                };
                
                // 等待400毫秒让数据加载完成
                const start = Date.now();
                while (Date.now() - start < 400) {
                    // 空循环等待
                }
                
                // 检查window对象中的数据
                const windowCheck = {
                    v_data: {
                        exists: typeof window.v_data !== 'undefined',
                        type: typeof window.v_data,
                        value: window.v_data
                    },
                    newdata: {
                        exists: typeof window.newdata !== 'undefined',
                        type: typeof window.newdata,
                        value: window.newdata
                    },
                    h_data: {
                        exists: typeof window.h_data !== 'undefined',
                        type: typeof window.h_data,
                        value: window.h_data
                    },
                    vs_data: {
                        exists: typeof window.vs_data !== 'undefined',
                        type: typeof window.vs_data,
                        value: window.vs_data
                    },
                    history_data: {
                        exists: typeof window.history_data !== 'undefined',
                        type: typeof window.history_data,
                        value: window.history_data
                    }
                };
                
                // 将window检查结果添加到返回数据中
                data.windowCheck = windowCheck;
                
                data.awayData = convertObject(window.newdata); // 客队近期数据
                data.homeData = convertObject(window.h_data); // 主队近期数据
                data.historyData = convertObject(window.v_data); // 历史交锋数据
                
                return data;
            }''')
        else:
            # 获取比赛信息
            data = await page.evaluate('''() => {
                const data = {};
                data.arrLeague = window.arrLeague;
                data.arrTeam = window.arrTeam;
                data.jh = window.jh;
                return data;
            }''')
        
        # 关闭浏览器
        await browser.close()
        
        return data
    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None

if __name__ == "__main__":
    # 测试代码
    test_url = "http://zq.titan007.com/cn/League/2023-2024/36.html"
    data = asyncio.get_event_loop().run_until_complete(get_all_match(test_url))
    print(json.dumps(data, ensure_ascii=False, indent=2))
