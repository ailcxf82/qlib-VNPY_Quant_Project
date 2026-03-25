from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class TushareClientManager:
    _instance: Optional['TushareClientManager'] = None
    _lock: threading.Lock = threading.Lock()
    _ts_client: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_client(cls) -> Optional[Any]:
        if cls._ts_client is None:
            with cls._lock:
                if cls._ts_client is None:
                    try:
                        import sys
                        project_root = Path(__file__).parent.parent
                        msa_path = str(project_root / "backtest" / "msa")
                        if msa_path not in sys.path:
                            sys.path.insert(0, msa_path)
                        from tushare_client import TushareClient
                        cls._ts_client = TushareClient.try_create()
                        if cls._ts_client:
                            logger.info("TushareClient 初始化成功")
                    except Exception as e:
                        logger.warning(f"TushareClient 初始化失败: {e}")
        return cls._ts_client
    
    @classmethod
    def reset(cls):
        with cls._lock:
            cls._ts_client = None
            logger.info("TushareClient 已重置")


def get_ts_client() -> Optional[Any]:
    return TushareClientManager.get_client()
