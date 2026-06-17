"""
配置加载工具，负责解析 YAML 并提供统一的字典对象。
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    参数
    ----
    path : str
        文件相对或绝对路径。
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"配置文件路径无效: {path!r}")

    raw = path
    path = path.strip()

    def _candidates(p: str) -> list[str]:
        cands: list[str] = []
        # 原样
        cands.append(p)
        # 兼容：不带扩展名
        base, ext = os.path.splitext(p)
        if ext == "":
            cands.append(p + ".yaml")
            cands.append(p + ".yml")
        # 兼容：当工作目录不在项目根目录时，按“项目根目录”回退解析一次
        # project_root = utils/.. （即仓库根）
        if not os.path.isabs(p):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            cands.append(os.path.join(project_root, p))
            if ext == "":
                cands.append(os.path.join(project_root, p + ".yaml"))
                cands.append(os.path.join(project_root, p + ".yml"))
        # 去重但保持顺序
        seen = set()
        out = []
        for x in cands:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    chosen = None
    for cand in _candidates(path):
        if os.path.exists(cand):
            chosen = cand
            break
    if chosen is None:
        raise FileNotFoundError(f"配置文件不存在: {raw}（尝试路径: {_candidates(path)}）")

    with open(chosen, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


