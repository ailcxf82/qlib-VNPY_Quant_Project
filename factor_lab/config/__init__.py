"""factor_lab 配置加载器。

阶段 F.1 / F.3 把两类领域知识从 Python 硬编码迁出到 YAML：

* ``factor_families.yaml`` —— 家族关键词 → 家族标签映射（F.1）
* ``rag_constitution.yaml`` —— RD-Agent RAG 静态宪法（F.3）

加载策略：**YAML 主源 + Python 兜底**。YAML 缺失或校验失败时回退到内置默认值，
同时 ``logger.warning`` 提示研究员修复。这样确保 RD-Agent / aggregator 永远能
启动。
"""

from __future__ import annotations

from factor_lab.config.families import (
    DEFAULT_FAMILY_RULES,
    FamilyRule,
    classify_family,
    get_family_rules,
    load_family_rules,
)

__all__ = [
    "DEFAULT_FAMILY_RULES",
    "FamilyRule",
    "classify_family",
    "get_family_rules",
    "load_family_rules",
]
