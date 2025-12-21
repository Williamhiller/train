/**
 * Created by Williamhiler on 2016/11/22.
 */

var phantom = require('phantom');
var Q = require("q");
var _ph, _page;

module.exports = function (url, type) {
    var deferred = Q.defer();

    var _url = url;

    phantom.create().then(function (ph) {
        _ph = ph;
        return _ph.createPage();
    }).then(function (page) {
        _page = page;
        _page.setting('userAgent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36');

        return _page.open(_url);
    }).then(function (status) {
        console.log(status);

        return _page.evaluate(function (type) {
            var data = {};
            // 解决phantomjs对象传递问题：先转为JSON字符串再转回对象
            var convertObject = function(obj) {
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
            if(type === 'odd'){
                // 详情数据
                data.game = convertObject(game);
                data.gameDetail = convertObject(gameDetail);
                return data;
            }
            if(type === 'analyze') {
                // 等待400毫秒让数据加载完成
                var start = Date.now();
                while (Date.now() - start < 400) {
                    // 空循环等待
                }
                
                // 简单直接的方式检查window对象中的数据
                var windowCheck = {
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
                
                // 将window检查结果添加到返回数据中，这样我们就能在外部看到了
                data.windowCheck = windowCheck;
                
                data.awayData = convertObject(window.newdata); // 客队近期数据
                data.homeData = convertObject(window.h_data); // 主队近期数据
                data.historyData = convertObject(window.v_data); // 历史交锋数据
                
                
                return data;
            }
            // 比赛信息
            data.arrLeague = arrLeague;
            data.arrTeam = arrTeam;
            data.jh = jh;

            return data;

        }, type);
    }).then(function (data) {
        deferred.resolve(data);

        _page.close();
        _ph.exit(0);
    }).catch(function (e) {
        console.log(e);
        deferred.reject(e);
    });

    return deferred.promise;
};



