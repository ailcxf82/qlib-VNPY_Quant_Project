#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日线股票投资监听系统 - 入口脚本

使用方法:
    # 启动定时调度器（每日收盘后自动运行）
    python run_monitor.py --mode scheduler
    
    # 手动运行一次监听任务
    python run_monitor.py --mode once
    
    # 启动 Web API 服务
    python run_monitor.py --mode api
    
    # 查看当前状态
    python run_monitor.py --mode status
    
    # 重置持仓
    python run_monitor.py --mode reset
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "data" / "monitor" / "monitor.log",
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)


def setup_data_dir():
    data_dir = project_root / "data" / "monitor"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def run_scheduler(config_path: str):
    from monitor.scheduler import MonitorScheduler
    
    logger.info("=" * 60)
    logger.info("日线监听系统 - 定时调度模式")
    logger.info("=" * 60)
    
    scheduler = MonitorScheduler(config_path)
    scheduler.start()
    
    logger.info(f"调度器已启动，执行时间: {scheduler.schedule_time}")
    logger.info("按 Ctrl+C 退出...")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("接收到退出信号，停止调度器...")
        scheduler.stop()
        logger.info("调度器已停止")


def run_once(config_path: str, date: str = None):
    from monitor.scheduler import MonitorScheduler
    
    logger.info("=" * 60)
    logger.info("日线监听系统 - 单次执行模式")
    logger.info("=" * 60)
    
    scheduler = MonitorScheduler(config_path)
    result = scheduler.run_once(date)
    
    if result.get("success"):
        logger.info("=" * 60)
        logger.info("监听任务执行成功")
        portfolio = result.get("portfolio", {})
        logger.info(f"总资产: ¥{portfolio.get('total_assets', 0):,.0f}")
        logger.info(f"总收益: {portfolio.get('total_return', 0):.2%}")
        logger.info(f"持仓数量: {portfolio.get('position_count', 0)}")
    else:
        logger.error(f"监听任务执行失败: {result.get('error', '未知错误')}")
    
    return result


def run_api(config_path: str, port: int = 5000):
    logger.info("=" * 60)
    logger.info("日线监听系统 - API 服务模式")
    logger.info("=" * 60)
    
    from api.predict_api import app, get_monitor_scheduler
    
    scheduler = get_monitor_scheduler()
    if scheduler:
        logger.info("监听系统组件已初始化")
    
    logger.info(f"API 服务启动中...")
    logger.info(f"API 地址: http://127.0.0.1:{port}")
    logger.info(f"Web 界面: http://127.0.0.1:{port}/")
    logger.info("")
    logger.info("监听系统 API 端点:")
    logger.info(f"  - GET  /api/monitor/status    获取系统状态")
    logger.info(f"  - POST /api/monitor/run       手动运行监听任务")
    logger.info(f"  - POST /api/monitor/start     启动定时调度")
    logger.info(f"  - POST /api/monitor/stop      停止定时调度")
    logger.info(f"  - GET  /api/monitor/portfolio 获取持仓信息")
    logger.info(f"  - GET  /api/monitor/signals   获取信号列表")
    logger.info(f"  - GET  /api/monitor/sentiment 获取舆情分析")
    logger.info(f"  - POST /api/monitor/reset     重置持仓")
    
    app.run(host="0.0.0.0", port=port, debug=False)


def run_status(config_path: str):
    from monitor.scheduler import MonitorScheduler
    
    scheduler = MonitorScheduler(config_path)
    status = scheduler.get_status()
    
    print("\n" + "=" * 60)
    print("日线监听系统 - 当前状态")
    print("=" * 60)
    print(f"调度器状态: {'运行中' if status['running'] else '已停止'}")
    print(f"执行时间: {status['schedule_time']}")
    print(f"上次执行: {status['last_run_date'] or '未执行'}")
    
    if status.get("portfolio"):
        portfolio = status["portfolio"]
        print("\n账户状态:")
        print(f"  总资产: ¥{portfolio['total_assets']:,.0f}")
        print(f"  总收益: {portfolio['total_return']:.2%}")
        print(f"  现金: ¥{portfolio['cash']:,.0f}")
        print(f"  持仓数量: {portfolio['position_count']}")
        
        if portfolio.get("positions"):
            print("\n当前持仓:")
            for pos in portfolio["positions"][:10]:
                profit_emoji = "🟢" if pos["profit_pct"] >= 0 else "🔴"
                print(f"  {pos['code']} {pos['name']}: "
                      f"{profit_emoji} {pos['profit_pct']:.2%} ({pos['holding_days']}天)")
    print("=" * 60 + "\n")


def run_reset(config_path: str):
    from monitor.scheduler import MonitorScheduler
    
    scheduler = MonitorScheduler(config_path)
    scheduler.reset_positions()
    
    print("\n持仓已重置\n")


def main():
    parser = argparse.ArgumentParser(
        description="日线股票投资监听系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 启动定时调度器
    python run_monitor.py --mode scheduler
    
    # 手动运行一次
    python run_monitor.py --mode once
    
    # 指定日期运行
    python run_monitor.py --mode once --date 2024-01-15
    
    # 启动 API 服务
    python run_monitor.py --mode api --port 5000
    
    # 查看状态
    python run_monitor.py --mode status
    
    # 重置持仓
    python run_monitor.py --mode reset
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["scheduler", "once", "api", "status", "reset"],
        default="once",
        help="运行模式: scheduler(定时调度), once(单次执行), api(Web服务), status(查看状态), reset(重置持仓)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config/monitor.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="指定运行日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="API 服务端口"
    )
    
    args = parser.parse_args()
    
    setup_data_dir()
    
    if args.mode == "scheduler":
        run_scheduler(args.config)
    elif args.mode == "once":
        run_once(args.config, args.date)
    elif args.mode == "api":
        run_api(args.config, args.port)
    elif args.mode == "status":
        run_status(args.config)
    elif args.mode == "reset":
        run_reset(args.config)


if __name__ == "__main__":
    main()
