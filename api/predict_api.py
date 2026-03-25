"""
股票预测 API 服务
提供 REST API 接口获取模型预测结果（集成 MSA 投资策略）
扩展：日线监听系统 API
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request, render_template_string

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_monitor_scheduler = None
_monitor_initialized = False

# 导入 MSA 策略相关模块
try:
    from backtest.msa.filters import FilterConfig, apply_basic_filters
    from backtest.msa.prediction_loader import load_prediction_csv, topk
    from backtest.msa.tushare_client import TushareClient
    from backtest.msa.code_utils import rqalpha_to_tushare
    MSA_AVAILABLE = True
except Exception as e:
    logging.warning(f"MSA 模块导入失败: {e}")
    MSA_AVAILABLE = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTION_DIR = os.path.join(project_root, "data", "predictions")
STOCK_NAMES = {
    "000001": "平安银行", "000002": "万科A", "000063": "中兴通讯",
    "000333": "美的集团", "000651": "格力电器", "000725": "京东方A",
    "000858": "五粮液", "002415": "海康威视", "002594": "比亚迪",
    "300059": "东方财富", "300750": "宁德时代", "600000": "浦发银行",
    "600036": "招商银行", "600519": "贵州茅台", "600887": "伊利股份",
    "601318": "中国平安", "601398": "工商银行", "601939": "建设银行",
    "000100": "TCL科技", "000157": "中联重科", "000166": "申万宏源",
    "000301": "东方盛虹", "000338": "潍柴动力",
    "000425": "徐工机械", "000625": "长安汽车",
    "000703": "建发股份", "000708": "中信特钢",
    "000768": "中航西飞", "000776": "广发证券",
    "000876": "新希望", "000977": "浪潮信息",
    "001979": "招商蛇口", "002001": "新和成", "002027": "分众传媒",
    "002049": "紫光国微", "002050": "三花智控", "002129": "中环股份",
    "002142": "宁波银行", "002230": "科大讯飞", "002241": "歌尔股份",
    "002271": "东方雨虹", "002304": "洋河股份", "002352": "顺丰控股",
    "002410": "广联达", "002475": "立讯精密",
    "002600": "领益智造", "002648": "卫星化学",
    "002714": "牧原股份", "002812": "恩捷股份", "002841": "视源股份",
    "003816": "中国广核", "300014": "亿纬锂能", "300033": "同花顺",
    "300124": "汇川技术", "300142": "沃森生物",
    "300274": "阳光电源", "300308": "中际旭创", "300347": "泰格医药",
    "300390": "天华超净", "300413": "芒果超媒", "300433": "蓝思科技",
    "300450": "先导智能", "300496": "中科创达", "300498": "温氏股份",
    "300502": "新易盛", "300760": "迈瑞医疗",
    "300782": "卓胜微", "300896": "爱美客",
    "600009": "上海机场", "600010": "包钢股份", "600011": "华能国际",
    "600015": "华夏银行", "600016": "民生银行", "600018": "上港集团",
    "600019": "宝钢股份", "600025": "华能水电", "600028": "中国石化",
    "600029": "南方航空", "600030": "中信证券", "600031": "三一重工",
    "600048": "保利发展", "600050": "中国联通",
    "600061": "国投资本", "600085": "同仁堂", "600104": "上汽集团",
    "600111": "北方稀土", "600132": "重庆啤酒", "600150": "中国船舶",
    "600176": "中国巨石", "600183": "生益科技", "600196": "复星医药",
    "600219": "南山铝业", "600233": "圆通速递", "600276": "恒瑞医药",
    "600309": "万华化学", "600332": "白云山", "600346": "恒力石化",
    "600352": "浙江龙盛", "600362": "江西铜业", "600383": "金地集团",
    "600406": "国电南瑞", "600438": "通威股份", "600486": "扬农化工",
    "600489": "中金黄金", "600547": "山东黄金",
    "600570": "恒生电子", "600585": "海螺水泥", "600588": "用友网络",
    "600660": "福耀玻璃", "600674": "川投能源",
    "600690": "海尔智家", "600703": "三安光电", "600745": "闻泰科技",
    "600809": "山西汾酒", "600837": "海通证券", "600845": "宝信软件",
    "600875": "东方电气", "600886": "国投电力",
    "600893": "航发动力", "600900": "长江电力", "600918": "中泰证券",
    "600926": "杭州银行", "600941": "中国移动", "600958": "东方证券",
    "600989": "宝丰能源", "601012": "隆基绿能", "601066": "中信建投",
    "601088": "中国神华", "601111": "中国国航", "601127": "赛力斯",
    "601138": "工业富联", "601166": "兴业银行", "601225": "陕西煤业",
    "601236": "红塔证券", "601238": "广汽集团", "601288": "农业银行",
    "601328": "交通银行", "601336": "新华保险",
    "601390": "中国中铁", "601601": "中国太保",
    "601628": "中国人寿", "601633": "长城汽车", "601668": "中国建筑",
    "601669": "中国电建", "601688": "华泰证券", "601728": "中国电信",
    "601766": "中国中车", "601788": "光大证券", "601800": "中国交建",
    "601818": "光大银行", "601857": "中国石油", "601877": "正泰电器",
    "601888": "中国中免", "601899": "紫金矿业", "601901": "方正证券",
    "601919": "中远海控", "601988": "中国银行",
    "601989": "中国重工", "601995": "中金公司", "601998": "中信银行",
    "603019": "中科曙光", "600160": "巨化股份", "603259": "药明康德",
    "603260": "合盛硅业", "603288": "海天味业", "603296": "华勤技术",
    "603501": "韦尔股份", "603799": "华友钴业", "603833": "欧派家居",
    "603899": "晨光股份", "603986": "兆易创新", "603993": "洛阳钼业",
    "688008": "澜起科技", "688012": "中微公司", "688041": "海光信息",
    "688111": "金山办公", "688126": "沪硅产业", "688169": "石头科技",
    "688187": "时代电气", "688223": "晶科能源", "688256": "寒武纪",
    "688303": "大全能源", "688396": "华润微", "688472": "德科立",
    "688599": "天合光能", "688981": "中芯国际"
}

_stock_name_cache = {}

def get_stock_name(code: str) -> str:
    code_num = str(code).replace("SH", "").replace("SZ", "").zfill(6)
    
    if code_num in _stock_name_cache:
        return _stock_name_cache[code_num]
    
    if code_num in STOCK_NAMES:
        return STOCK_NAMES[code_num]
    
    try:
        ts = TushareClient.try_create()
        if ts:
            sb = ts.stock_basic()
            sb_map = sb.set_index('ts_code').to_dict(orient='index')
            
            for suffix in ['.SZ', '.SH']:
                ts_code = code_num + suffix
                if ts_code in sb_map:
                    name = sb_map[ts_code].get('name', code_num)
                    _stock_name_cache[code_num] = name
                    return name
    except Exception:
        pass
    
    return code_num


def load_predictions(pool_name: str) -> Optional[pd.DataFrame]:
    pred_file = os.path.join(PREDICTION_DIR, f"pred_{pool_name}.csv")
    if not os.path.exists(pred_file):
        return None
    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def apply_msa_strategy(pool_name: str, date: str) -> List[str]:
    """应用 MSA 投资策略选择股票"""
    if not MSA_AVAILABLE:
        logger.warning("MSA 模块不可用，使用默认排序")
        return []
    
    try:
        df = load_predictions(pool_name)
        if df is None:
            return []
        
        target_date = pd.to_datetime(date)
        df = df[df["datetime"] == target_date]
        if df.empty:
            return []
        
        # 按预测值排序
        if "final" in df.columns:
            df = df.sort_values("final", ascending=False)
            pred_col = "final"
        elif "lgb" in df.columns:
            df = df.sort_values("lgb", ascending=False)
            pred_col = "lgb"
        else:
            df = df.sort_values(df.columns[2], ascending=False)
            pred_col = df.columns[2]
        
        # 应用策略过滤
        from backtest.msa.code_utils import qlib_to_rqalpha
        rq_codes = [qlib_to_rqalpha(str(code)) for code in df["instrument"].head(20)]
        
        # 初始化 Tushare 客户端
        ts = TushareClient.try_create()
        
        # 策略1：小市值策略（CSI101）
        if pool_name == "csi101":
            filter_cfg = FilterConfig(
                exclude_kcb_bj=True,
                exclude_st=True,
                min_list_days=360,
                pb_min=0,
                pb_max=None
            )
            filtered = apply_basic_filters(rq_codes, target_date, filter_cfg, ts)
            return filtered[:6]
        
        # 策略2：低估值策略（CSI300）
        elif pool_name == "csi300":
            filter_cfg = FilterConfig(
                exclude_kcb_bj=True,
                exclude_st=True,
                min_list_days=360,
                pb_min=0,
                pb_max=None
            )
            filtered = apply_basic_filters(rq_codes, target_date, filter_cfg, ts)
            return filtered[:4]
        
        return rq_codes[:10]
        
    except Exception as e:
        logger.error(f"MSA 策略应用失败: {e}")
        return []


def get_latest_predictions(pool_name: str, date: str = None, use_strategy: bool = True) -> Dict:
    df = load_predictions(pool_name)
    if df is None or df.empty:
        raise FileNotFoundError(f"未找到股票池 {pool_name} 的预测文件，请先运行 python run_predict.py")
    
    if date:
        target_date = pd.to_datetime(date)
        df = df[df["datetime"] == target_date]
        if df.empty:
            all_dates = sorted(load_predictions(pool_name)["datetime"].unique())
            latest_date = all_dates[-1]
            raise ValueError(f"日期 {date} 无预测数据，可用日期: {latest_date.date()}")
    else:
        all_dates = sorted(df["datetime"].unique())
        target_date = all_dates[-1]
        df = df[df["datetime"] == target_date]
    
    if "final" in df.columns:
        pred_col = "final"
    elif "lgb" in df.columns:
        pred_col = "lgb"
    else:
        pred_col = df.columns[2]
    
    df = df.sort_values(pred_col, ascending=False)
    
    model_weights = {}
    for col in ["lgb", "gru", "mlp", "stack", "qlib_ensemble"]:
        if col in df.columns:
            model_weights[col] = round(1.0 / len([c for c in ["lgb", "gru", "mlp", "stack", "qlib_ensemble"] if c in df.columns]), 4)
    
    # 应用 MSA 策略
    strategy_picks = []
    filtered_codes = []
    
    if use_strategy and MSA_AVAILABLE:
        strategy_picks = apply_msa_strategy(pool_name, str(target_date.date()))
        
        # 获取所有股票的过滤结果
        from backtest.msa.code_utils import qlib_to_rqalpha
        all_rq_codes = [qlib_to_rqalpha(str(code)) for code in df["instrument"]]
        
        # 初始化 Tushare 客户端
        ts = TushareClient.try_create()
        
        # 应用ST过滤
        filter_cfg = FilterConfig(
            exclude_kcb_bj=True,
            exclude_st=True,
            min_list_days=360,
            pb_min=0,
            pb_max=None
        )
        
        filtered_rq_codes = apply_basic_filters(all_rq_codes, target_date, filter_cfg, ts)
        from backtest.msa.code_utils import rqalpha_to_pure_code
        filtered_codes = [rqalpha_to_pure_code(code) for code in filtered_rq_codes]
    
    predictions = []
    code_to_pred = {str(row["instrument"]).zfill(6): float(row[pred_col]) for _, row in df.iterrows()}
    
    def format_code(code):
        return str(code).zfill(6)
    
    # 优先显示策略选择的股票
    if strategy_picks:
        from backtest.msa.code_utils import rqalpha_to_pure_code
        for rq_code in strategy_picks:
            pure_code = rqalpha_to_pure_code(rq_code)
            pure_code_formatted = format_code(pure_code)
            if pure_code_formatted in code_to_pred:
                predictions.append({
                    "code": pure_code_formatted,
                    "name": get_stock_name(pure_code),
                    "prediction": round(code_to_pred[pure_code_formatted], 4),
                    "date": str(target_date.date()),
                    "strategy": True
                })
        # 补充其他推荐股票（只显示经过ST过滤的股票）
        strategy_pure_codes = [format_code(rqalpha_to_pure_code(code)) for code in strategy_picks]
        remaining = [code for code in code_to_pred if code not in strategy_pure_codes and code in filtered_codes]
        for code in remaining[:50 - len(strategy_picks)]:
            predictions.append({
                "code": code,
                "name": get_stock_name(code),
                "prediction": round(code_to_pred[code], 4),
                "date": str(target_date.date()),
                "strategy": False
            })
    else:
        # 没有策略结果时，按预测值排序（只显示经过ST过滤的股票）
        for _, row in df.head(50).iterrows():
            code = format_code(row["instrument"])
            if code in filtered_codes or not filtered_codes:
                predictions.append({
                    "code": code,
                    "name": get_stock_name(str(row["instrument"])),
                    "prediction": round(float(row[pred_col]), 4),
                    "date": str(row["datetime"].date()) if hasattr(row["datetime"], 'date') else str(row["datetime"]),
                    "strategy": False
                })
    
    return {
        "pool_name": pool_name,
        "prediction_date": str(target_date.date()) if 'target_date' in dir() else str(df["datetime"].iloc[0].date()),
        "generated_at": datetime.now().isoformat(),
        "model_weights": model_weights,
        "strategy_applied": use_strategy and MSA_AVAILABLE,
        "strategy_picks_count": len(strategy_picks),
        "predictions": predictions
    }


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        pool_name = request.args.get('pool', 'csi300')
        date = request.args.get('start_date') or request.args.get('date')
        use_strategy = request.args.get('strategy', 'true').lower() == 'true'
        
        result = get_latest_predictions(pool_name, date, use_strategy)
        return jsonify({"success": True, "data": result})
        
    except FileNotFoundError as e:
        logger.error(f"预测文件不存在: {e}")
        return jsonify({"success": False, "error": str(e)}), 404
    except ValueError as e:
        logger.error(f"日期错误: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/pools', methods=['GET'])
def get_pools():
    pools = []
    if os.path.exists(PREDICTION_DIR):
        for f in os.listdir(PREDICTION_DIR):
            if f.startswith("pred_") and f.endswith(".csv"):
                pool_name = f.replace("pred_", "").replace(".csv", "")
                pools.append(pool_name)
    if not pools:
        pools = ["csi300", "csi101"]
    return jsonify({"success": True, "data": pools})


@app.route('/api/dates', methods=['GET'])
def get_dates():
    try:
        pool_name = request.args.get('pool', 'csi300')
        df = load_predictions(pool_name)
        if df is None:
            return jsonify({"success": False, "error": "未找到预测文件"}), 404
        dates = sorted(df["datetime"].dt.strftime("%Y-%m-%d").unique().tolist())
        return jsonify({"success": True, "data": dates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat(), "msa_available": MSA_AVAILABLE})


def get_monitor_scheduler():
    global _monitor_scheduler, _monitor_initialized
    if not _monitor_initialized:
        try:
            from monitor.scheduler import MonitorScheduler
            _monitor_scheduler = MonitorScheduler()
            _monitor_initialized = True
        except Exception as e:
            logger.error(f"初始化监听系统失败: {e}")
    return _monitor_scheduler


@app.route('/api/monitor/status', methods=['GET'])
def monitor_status():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        status = scheduler.get_status()
        return jsonify({"success": True, "data": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/run', methods=['POST'])
def monitor_run():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        data = request.get_json() or {}
        date = data.get('date')
        
        result = scheduler.run_once(date)
        return jsonify({"success": result.get("success", False), "data": result})
    except Exception as e:
        logger.error(f"运行监听任务失败: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/start', methods=['POST'])
def monitor_start():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        scheduler.start()
        return jsonify({"success": True, "message": "监听系统已启动"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/stop', methods=['POST'])
def monitor_stop():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        scheduler.stop()
        return jsonify({"success": True, "message": "监听系统已停止"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/portfolio', methods=['GET'])
def monitor_portfolio():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        portfolio = scheduler.position_tracker.get_portfolio_summary()
        return jsonify({"success": True, "data": portfolio})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/signals', methods=['GET'])
def monitor_signals():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        date = request.args.get('date')
        signals = scheduler.signal_engine.generate_all_signals(date)
        
        result = {}
        for pool, sigs in signals.items():
            result[pool] = [s.to_dict() for s in sigs]
        
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/sentiment', methods=['GET'])
def monitor_sentiment():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        result = scheduler.sentiment_analyzer.analyze(force_refresh)
        return jsonify({"success": True, "data": result.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/monitor/reset', methods=['POST'])
def monitor_reset():
    try:
        scheduler = get_monitor_scheduler()
        if scheduler is None:
            return jsonify({"success": False, "error": "监听系统未初始化"}), 500
        
        scheduler.reset_positions()
        return jsonify({"success": True, "message": "持仓已重置"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票预测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .card { background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); padding: 24px; margin-bottom: 20px; }
        .form-row { display: flex; gap: 15px; flex-wrap: wrap; align-items: flex-end; }
        .form-group { flex: 1; min-width: 200px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .form-group select, .form-group input { width: 100%; padding: 12px 16px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }
        .form-group select:focus, .form-group input:focus { outline: none; border-color: #667eea; }
        .btn { padding: 12px 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loading { display: none; text-align: center; padding: 40px; }
        .loading.active { display: block; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .results { display: none; }
        .results.active { display: block; }
        .info-bar { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px; }
        .info-item { text-align: center; }
        .info-item .label { font-size: 12px; color: #666; margin-bottom: 4px; }
        .info-item .value { font-size: 18px; font-weight: 600; color: #333; }
        .weights { display: flex; flex-wrap: wrap; gap: 10px; }
        .weight-tag { padding: 6px 12px; background: #e8f4f8; border-radius: 20px; font-size: 14px; }
        .weight-tag span { font-weight: 600; color: #667eea; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #333; position: sticky; top: 0; }
        tr:hover { background: #f8f9fa; }
        .rank { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; font-size: 14px; }
        .rank.top3 { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
        .rank.strategy { background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); }
        .prediction-bar { height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }
        .prediction-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 4px; }
        .error { display: none; padding: 20px; background: #fee; border: 1px solid #fcc; border-radius: 8px; color: #c00; }
        .error.active { display: block; }
        .tip { background: #e3f2fd; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; color: #1565c0; }
        .strategy-badge { display: inline-block; padding: 2px 8px; background: #4caf50; color: white; border-radius: 12px; font-size: 12px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 股票预测系统</h1>
            <p>基于机器学习的智能选股推荐（集成 MSA 投资策略）</p>
        </div>
        
        <div class="card">
            <div class="tip">
                💡 提示：本系统读取已生成的预测文件。如需最新预测，请先运行 <code>python run_predict.py</code>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>股票池</label>
                    <select id="poolSelect"></select>
                </div>
                <div class="form-group">
                    <label>预测日期（留空则显示最新）</label>
                    <select id="dateSelect">
                        <option value="">最新日期</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>应用投资策略</label>
                    <select id="strategySelect">
                        <option value="true">是（推荐）</option>
                        <option value="false">否</option>
                    </select>
                </div>
                <button class="btn" id="predictBtn" onclick="runPrediction()">查询预测</button>
            </div>
        </div>
        
        <div class="card loading" id="loading">
            <div class="spinner"></div>
            <p>正在加载预测结果...</p>
        </div>
        
        <div class="card error" id="error"></div>
        
        <div class="card results" id="results">
            <div class="info-bar">
                <div class="info-item">
                    <div class="label">股票池</div>
                    <div class="value" id="infoPool">-</div>
                </div>
                <div class="info-item">
                    <div class="label">预测日期</div>
                    <div class="value" id="infoDate">-</div>
                </div>
                <div class="info-item">
                    <div class="label">查询时间</div>
                    <div class="value" id="infoTime">-</div>
                </div>
                <div class="info-item">
                    <div class="label">推荐数量</div>
                    <div class="value" id="infoCount">-</div>
                </div>
                <div class="info-item">
                    <div class="label">策略推荐</div>
                    <div class="value" id="infoStrategy">-</div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="margin-bottom: 10px;">模型权重</h3>
                <div class="weights" id="weights"></div>
            </div>
            
            <h3 style="margin-bottom: 10px;">推荐股票 (按预测值排序)</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>股票代码</th>
                            <th>股票名称</th>
                            <th>预测值</th>
                            <th>预测强度</th>
                            <th>策略推荐</th>
                        </tr>
                    </thead>
                    <tbody id="stockTable"></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        async function loadPools() {
            try {
                const response = await fetch('/api/pools');
                const data = await response.json();
                if (data.success) {
                    const select = document.getElementById('poolSelect');
                    select.innerHTML = '';
                    data.data.forEach(pool => {
                        const option = document.createElement('option');
                        option.value = pool;
                        option.textContent = pool === 'csi300' ? '沪深300' : pool === 'csi101' ? '中小综指' : pool;
                        select.appendChild(option);
                    });
                    loadDates();
                }
            } catch (e) {
                console.error('加载股票池失败:', e);
            }
        }
        
        async function loadDates() {
            const pool = document.getElementById('poolSelect').value;
            try {
                const response = await fetch(`/api/dates?pool=${pool}`);
                const data = await response.json();
                if (data.success) {
                    const select = document.getElementById('dateSelect');
                    select.innerHTML = '<option value="">最新日期</option>';
                    data.data.reverse().forEach(date => {
                        const option = document.createElement('option');
                        option.value = date;
                        option.textContent = date;
                        select.appendChild(option);
                    });
                }
            } catch (e) {
                console.error('加载日期失败:', e);
            }
        }
        
        document.getElementById('poolSelect').addEventListener('change', loadDates);
        
        async function runPrediction() {
            const pool = document.getElementById('poolSelect').value;
            const date = document.getElementById('dateSelect').value;
            const strategy = document.getElementById('strategySelect').value;
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');
            document.getElementById('error').classList.remove('active');
            document.getElementById('predictBtn').disabled = true;
            
            try {
                const params = new URLSearchParams({ pool, strategy });
                if (date) params.append('date', date);
                const response = await fetch(`/api/predict?${params}`);
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.data);
                } else {
                    throw new Error(result.error || '查询失败');
                }
            } catch (e) {
                document.getElementById('error').textContent = '错误: ' + e.message;
                document.getElementById('error').classList.add('active');
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('predictBtn').disabled = false;
            }
        }
        
        function displayResults(data) {
            document.getElementById('infoPool').textContent = data.pool_name === 'csi300' ? '沪深300' : data.pool_name;
            document.getElementById('infoDate').textContent = data.prediction_date;
            document.getElementById('infoTime').textContent = new Date(data.generated_at).toLocaleString('zh-CN');
            document.getElementById('infoCount').textContent = data.predictions.length + ' 只';
            document.getElementById('infoStrategy').textContent = data.strategy_applied ? `是 (${data.strategy_picks_count}只)` : '否';
            
            const weightsDiv = document.getElementById('weights');
            weightsDiv.innerHTML = '';
            for (const [model, weight] of Object.entries(data.model_weights || {})) {
                const tag = document.createElement('span');
                tag.className = 'weight-tag';
                tag.innerHTML = `${model}: <span>${(weight * 100).toFixed(1)}%</span>`;
                weightsDiv.appendChild(tag);
            }
            
            const tbody = document.getElementById('stockTable');
            tbody.innerHTML = '';
            data.predictions.forEach((stock, index) => {
                const tr = document.createElement('tr');
                let rankClass = 'rank';
                if (index < 3) rankClass += ' top3';
                if (stock.strategy) rankClass += ' strategy';
                const barWidth = (stock.prediction * 100).toFixed(0);
                tr.innerHTML = `
                    <td><span class="${rankClass}">${index + 1}</span></td>
                    <td><code>${stock.code}</code></td>
                    <td><strong>${stock.name}</strong></td>
                    <td>${stock.prediction.toFixed(4)}</td>
                    <td style="width: 200px;">
                        <div class="prediction-bar">
                            <div class="prediction-fill" style="width: ${barWidth}%"></div>
                        </div>
                    </td>
                    <td>${stock.strategy ? '<span class="strategy-badge">策略推荐</span>' : '-'}</td>
                `;
                tbody.appendChild(tr);
            });
            
            document.getElementById('results').classList.add('active');
        }
        
        loadPools();
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    print("=" * 60)
    print("股票预测 API 服务启动中...")
    print("=" * 60)
    print(f"API 地址: http://127.0.0.1:5000")
    print(f"Web 界面: http://127.0.0.1:5000/")
    print(f"预测接口: http://127.0.0.1:5000/api/predict")
    print(f"股票池列表: http://127.0.0.1:5000/api/pools")
    print(f"MSA 策略: {'可用' if MSA_AVAILABLE else '不可用'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
