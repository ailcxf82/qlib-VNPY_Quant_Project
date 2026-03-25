"""
Strategy Store - 策略存储层

提供策略配置、优化结果、回测结果的持久化存储。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from monitor.strategy_library.strategy_base import StrategyConfig, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecord:
    id: int
    strategy_id: str
    optimization_date: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "optimization_date": self.optimization_date,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "execution_time": self.execution_time,
        }


@dataclass
class StrategyRanking:
    rank: int
    strategy_id: str
    strategy_name: str
    category: str
    score: float
    ranking_date: str
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "category": self.category,
            "score": self.score,
            "ranking_date": self.ranking_date,
            "metrics": self.metrics,
        }


class StrategyStore:
    DEFAULT_DB_PATH = Path("data/strategy_library/strategies.db")
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    version TEXT NOT NULL,
                    config_yaml TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    optimization_date TIMESTAMP NOT NULL,
                    best_params TEXT NOT NULL,
                    best_score REAL NOT NULL,
                    all_results TEXT,
                    execution_time REAL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    backtest_date TIMESTAMP NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    annual_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    sortino_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    avg_holding_days REAL NOT NULL,
                    in_sample INTEGER NOT NULL DEFAULT 1,
                    metrics TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ranking_date TIMESTAMP NOT NULL,
                    strategy_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    score REAL NOT NULL,
                    metrics TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_strategy 
                ON optimization_results(strategy_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_strategy 
                ON backtest_results(strategy_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ranking_date 
                ON strategy_rankings(ranking_date)
            """)
            
            conn.commit()
        
        logger.info(f"策略存储初始化完成: {self.db_path}")
    
    def save_strategy(self, config: StrategyConfig) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO strategies (id, name, category, version, config_yaml, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    config.id,
                    config.name,
                    config.category,
                    config.version,
                    json.dumps(config.to_dict(), ensure_ascii=False),
                ))
                
                conn.commit()
            
            logger.debug(f"保存策略配置: {config.id}")
            return True
        except Exception as e:
            logger.error(f"保存策略配置失败 {config.id}: {e}")
            return False
    
    def load_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT config_yaml FROM strategies WHERE id = ?
                """, (strategy_id,))
                
                row = cursor.fetchone()
                
                if row:
                    return StrategyConfig.from_dict(json.loads(row[0]))
                
                return None
        except Exception as e:
            logger.error(f"加载策略配置失败 {strategy_id}: {e}")
            return None
    
    def list_strategies(self, category: Optional[str] = None) -> List[StrategyConfig]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if category:
                    cursor.execute("""
                        SELECT config_yaml FROM strategies WHERE category = ?
                    """, (category,))
                else:
                    cursor.execute("SELECT config_yaml FROM strategies")
                
                configs = []
                for row in cursor.fetchall():
                    configs.append(StrategyConfig.from_dict(json.loads(row[0])))
                
                return configs
        except Exception as e:
            logger.error(f"列出策略失败: {e}")
            return []
    
    def save_backtest_result(self, result: BacktestResult) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO backtest_results (
                        strategy_id, params_hash, backtest_date,
                        start_date, end_date, initial_capital, final_capital,
                        total_return, annual_return, sharpe_ratio, sortino_ratio,
                        max_drawdown, win_rate, profit_factor,
                        total_trades, winning_trades, losing_trades, avg_holding_days,
                        in_sample, metrics
                    ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.strategy_id,
                    result.get_params_hash(),
                    result.start_date,
                    result.end_date,
                    result.initial_capital,
                    result.final_capital,
                    result.total_return,
                    result.annual_return,
                    result.sharpe_ratio,
                    result.sortino_ratio,
                    result.max_drawdown,
                    result.win_rate,
                    result.profit_factor,
                    result.total_trades,
                    result.winning_trades,
                    result.losing_trades,
                    result.avg_holding_days,
                    1 if result.in_sample else 0,
                    json.dumps(result.metrics, ensure_ascii=False),
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
            
            logger.debug(f"保存回测结果: {result.strategy_id} (ID: {result_id})")
            return result_id
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
            return -1
    
    def get_best_result(
        self, 
        strategy_id: str,
        metric: str = "sharpe_ratio",
        in_sample: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = f"""
                    SELECT * FROM backtest_results 
                    WHERE strategy_id = ?
                """
                params = [strategy_id]
                
                if in_sample is not None:
                    query += " AND in_sample = ?"
                    params.append(1 if in_sample else 0)
                
                query += f" ORDER BY {metric} DESC LIMIT 1"
                
                cursor.execute(query, params)
                
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                
                return None
        except Exception as e:
            logger.error(f"获取最佳结果失败 {strategy_id}: {e}")
            return None
    
    def save_optimization_result(
        self,
        strategy_id: str,
        best_params: Dict[str, Any],
        best_score: float,
        all_results: List[Dict[str, Any]],
        execution_time: float
    ) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO optimization_results (
                        strategy_id, optimization_date, best_params, best_score,
                        all_results, execution_time
                    ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    json.dumps(best_params, ensure_ascii=False),
                    best_score,
                    json.dumps(all_results, ensure_ascii=False),
                    execution_time,
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
            
            logger.info(f"保存优化结果: {strategy_id} (得分: {best_score:.4f})")
            return result_id
        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
            return -1
    
    def get_latest_optimization(self, strategy_id: str) -> Optional[OptimizationRecord]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, strategy_id, optimization_date, best_params, best_score,
                           all_results, execution_time
                    FROM optimization_results
                    WHERE strategy_id = ?
                    ORDER BY optimization_date DESC
                    LIMIT 1
                """, (strategy_id,))
                
                row = cursor.fetchone()
                
                if row:
                    return OptimizationRecord(
                        id=row[0],
                        strategy_id=row[1],
                        optimization_date=row[2],
                        best_params=json.loads(row[3]),
                        best_score=row[4],
                        all_results=json.loads(row[5]) if row[5] else [],
                        execution_time=row[6] or 0.0,
                    )
                
                return None
        except Exception as e:
            logger.error(f"获取最新优化结果失败 {strategy_id}: {e}")
            return None
    
    def save_rankings(self, rankings: List[StrategyRanking]) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                ranking_date = datetime.now().isoformat()
                
                for ranking in rankings:
                    cursor.execute("""
                        INSERT INTO strategy_rankings (
                            ranking_date, strategy_id, strategy_name, category,
                            rank, score, metrics
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ranking_date,
                        ranking.strategy_id,
                        ranking.strategy_name,
                        ranking.category,
                        ranking.rank,
                        ranking.score,
                        json.dumps(ranking.metrics, ensure_ascii=False),
                    ))
                
                conn.commit()
            
            logger.info(f"保存策略排名: {len(rankings)} 个策略")
            return True
        except Exception as e:
            logger.error(f"保存策略排名失败: {e}")
            return False
    
    def get_latest_rankings(self, category: Optional[str] = None) -> List[StrategyRanking]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT MAX(ranking_date) FROM strategy_rankings
                """)
                
                latest_date = cursor.fetchone()[0]
                
                if not latest_date:
                    return []
                
                if category:
                    cursor.execute("""
                        SELECT strategy_id, strategy_name, category, rank, score, metrics, ranking_date
                        FROM strategy_rankings
                        WHERE ranking_date = ? AND category = ?
                        ORDER BY rank
                    """, (latest_date, category))
                else:
                    cursor.execute("""
                        SELECT strategy_id, strategy_name, category, rank, score, metrics, ranking_date
                        FROM strategy_rankings
                        WHERE ranking_date = ?
                        ORDER BY rank
                    """, (latest_date,))
                
                rankings = []
                for row in cursor.fetchall():
                    rankings.append(StrategyRanking(
                        strategy_id=row[0],
                        strategy_name=row[1],
                        category=row[2],
                        rank=row[3],
                        score=row[4],
                        metrics=json.loads(row[5]) if row[5] else {},
                        ranking_date=row[6],
                    ))
                
                return rankings
        except Exception as e:
            logger.error(f"获取最新排名失败: {e}")
            return []
    
    def cleanup_old_results(self, keep_days: int = 90) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - keep_days)
                cutoff_str = cutoff_date.isoformat()
                
                cursor.execute("""
                    DELETE FROM backtest_results 
                    WHERE backtest_date < ? AND id NOT IN (
                        SELECT MAX(id) FROM backtest_results GROUP BY strategy_id, params_hash
                    )
                """, (cutoff_str,))
                
                deleted = cursor.rowcount
                conn.commit()
            
            logger.info(f"清理过期回测结果: {deleted} 条")
            return deleted
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
            return 0
