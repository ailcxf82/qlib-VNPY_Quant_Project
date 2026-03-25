from __future__ import annotations


class MonitorError(Exception):
    """监听系统基础异常"""
    pass


class ConfigurationError(MonitorError):
    """配置错误"""
    pass


class DataNotFoundError(MonitorError):
    """数据未找到"""
    pass


class TushareError(MonitorError):
    """Tushare API 错误"""
    pass


class LLMError(MonitorError):
    """LLM API 错误"""
    pass


class NotificationError(MonitorError):
    """通知推送错误"""
    pass


class PositionError(MonitorError):
    """持仓操作错误"""
    pass


class TradeError(MonitorError):
    """交易执行错误"""
    pass


class SignalError(MonitorError):
    """信号生成错误"""
    pass


class NetworkError(MonitorError):
    """网络请求错误"""
    pass


class CacheError(MonitorError):
    """缓存操作错误"""
    pass
