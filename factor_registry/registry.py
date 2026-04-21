"""
``FactorRegistry`` —— L3 manifest + 证书副本的读写。

与 ``ParquetStore`` 的分工（见 ``docs/ARCHITECTURE_FACTOR_LAB.md`` §3/§5）：

* ``ParquetStore``   —— 管 ``factors_v<N>.parquet`` 物理文件（数据层）。
* ``FactorRegistry`` —— 管 ``manifest.json`` + ``data/certified/<factor_id>.json``
  (+ ``data/retired/<factor_id>.json``) 的元信息（索引层）。

Registry 是 L2 → L3 的**唯一入口**。任何想把因子推到 production 的调用路径，都必须：

1. 先拿到 ``CertifiedFactorRecord`` (C2)；
2. 调 ``FactorRegistry.register(certified, parquet_version=...)``；
3. 此时 registry 负责：
   * 校验 ``certified.decision == PASS``；
   * 把 C2 JSON 拷贝到 ``data/certified/<factor_id>.json``；
   * 构造 ``ProductionFactorRecord``；
   * 原子更新 ``manifest.json``。

读取路径（由 ``production_factor_loader`` 消费）：
* ``list_active()`` 返回所有 ``status == active`` 记录；
* ``list_by_version(v, only_active=True)`` 按 parquet 版本号过滤。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from factor_registry.schema import ProductionFactorRecord, ProductionStatus
from factor_validation.schema import CertifiedFactorRecord, Decision

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_CERTIFIED_DIR = "certified"
_RETIRED_DIR = "retired"
_SCHEMA_VERSION = 1


class RegistryError(RuntimeError):
    """``FactorRegistry`` 抛出的所有业务异常基类。"""


class FactorRegistry:
    """L3 manifest 读写 + 证书归档。

    Parameters
    ----------
    data_dir : Path
        registry 数据根目录，通常是 ``factor_registry/data/``。必须已存在并包含：

        * ``manifest.json``
        * ``certified/``
        * ``retired/``

        不存在会自动创建（适合测试临时目录）。
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / _CERTIFIED_DIR).mkdir(exist_ok=True)
        (self.data_dir / _RETIRED_DIR).mkdir(exist_ok=True)

        self.manifest_path = self.data_dir / _MANIFEST_FILENAME
        if not self.manifest_path.exists():
            self._write_manifest(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "last_updated_at": None,
                    "factors": [],
                }
            )
        self._manifest: dict = {}
        self._records: dict[str, ProductionFactorRecord] = {}
        self.load()

    # --------------------------------------------------------------- manifest IO

    def load(self) -> None:
        """从磁盘重载 manifest，覆盖内存缓存。"""
        raw = self._read_manifest()
        if not isinstance(raw, dict):
            raise RegistryError(f"manifest 根节点必须是对象: {self.manifest_path}")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise RegistryError(
                f"manifest schema_version 不匹配：期望 {_SCHEMA_VERSION}，"
                f"实际 {raw.get('schema_version')!r}"
            )
        factors = raw.get("factors")
        if not isinstance(factors, list):
            raise RegistryError("manifest.factors 必须为 list")

        records: dict[str, ProductionFactorRecord] = {}
        for item in factors:
            if not isinstance(item, dict):
                raise RegistryError(f"manifest.factors 元素必须为 dict: {item!r}")
            rec = ProductionFactorRecord.model_validate(item)
            if rec.factor_id in records:
                raise RegistryError(f"manifest 内 factor_id 重复: {rec.factor_id!r}")
            records[rec.factor_id] = rec
        self._manifest = raw
        self._records = records

    def save(self) -> None:
        """把内存中 records 回写 manifest.json（原子）。"""
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "factors": [
                self._record_to_dict(r)
                for r in sorted(
                    self._records.values(),
                    key=lambda r: (r.parquet_version, r.factor_id),
                )
            ],
        }
        self._write_manifest(payload)
        self._manifest = payload

    @staticmethod
    def _record_to_dict(rec: ProductionFactorRecord) -> dict:
        """Pydantic 序列化，把 Path / datetime / Enum 转成 JSON 安全形态。"""
        return json.loads(rec.model_dump_json())

    def _read_manifest(self) -> dict:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, payload: dict) -> None:
        """原子写入 manifest.json。"""
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=".manifest.", suffix=".tmp.json", dir=str(self.data_dir)
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
                f.write("\n")
            os.replace(tmp_path, self.manifest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ register

    def register(
        self,
        certified: CertifiedFactorRecord,
        *,
        parquet_version: int,
        tags: list[str] | None = None,
        now: datetime | None = None,
        allow_overwrite: bool = False,
    ) -> ProductionFactorRecord:
        """把一个已认证的因子推进 production。

        约束：

        * ``certified.decision == PASS``；否则报错（HOLD/FAIL 不应进入 L3）。
        * 同名 ``factor_id`` 已存在且 ``status == active`` 时：
          - ``allow_overwrite=False`` 报错；
          - ``allow_overwrite=True`` 直接覆盖旧记录。
        * 证书副本写入 ``data/certified/<factor_id>.json``（存在则覆盖）。
        """
        if not isinstance(certified, CertifiedFactorRecord):
            raise RegistryError(
                f"certified 必须是 CertifiedFactorRecord：{type(certified)!r}"
            )
        if certified.decision != Decision.PASS:
            raise RegistryError(
                f"只有 decision=PASS 的因子才能注册到 L3，"
                f"当前 {certified.factor_id}: decision={certified.decision.value}"
            )

        factor_id = certified.factor_id
        name = certified.candidate.name

        existing = self._records.get(factor_id)
        if existing is not None and existing.status == ProductionStatus.ACTIVE and not allow_overwrite:
            raise RegistryError(
                f"factor_id 已 active：{factor_id!r}；如需覆盖请传 allow_overwrite=True"
            )

        cert_rel = Path(_CERTIFIED_DIR) / f"{factor_id}.json"
        self._write_certified_copy(certified, self.data_dir / cert_rel)

        record = ProductionFactorRecord(
            factor_id=factor_id,
            name=name,
            status=ProductionStatus.ACTIVE,
            parquet_version=parquet_version,
            parquet_column=name,
            certificate_path=cert_rel,
            registered_at=(now or datetime.now(timezone.utc)),
            tags=list(tags or []),
        )
        self._records[factor_id] = record
        self.save()
        logger.info(
            "registry: 注册 %s (v%d, parquet_column=%s)",
            factor_id,
            parquet_version,
            name,
        )
        return record

    def _write_certified_copy(
        self, certified: CertifiedFactorRecord, path: Path
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            certified.model_dump_json(indent=2), encoding="utf-8"
        )

    # --------------------------------------------------------------------- retire

    def retire(
        self,
        factor_id: str,
        reason: str,
        *,
        when: datetime | None = None,
    ) -> ProductionFactorRecord:
        """将 active 因子迁为 retired；证书副本从 certified/ 搬到 retired/。"""
        if not reason or not reason.strip():
            raise RegistryError("retire reason 不能为空")
        rec = self._records.get(factor_id)
        if rec is None:
            raise RegistryError(f"factor_id 不存在：{factor_id!r}")
        if rec.status == ProductionStatus.RETIRED:
            raise RegistryError(f"factor_id 已 retired：{factor_id!r}")

        cert_src = self.data_dir / rec.certificate_path
        cert_dst_rel = Path(_RETIRED_DIR) / f"{factor_id}.json"
        cert_dst = self.data_dir / cert_dst_rel
        if cert_src.exists():
            cert_dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cert_src, cert_dst)

        retired = rec.model_copy(
            update={
                "status": ProductionStatus.RETIRED,
                "retired_at": (when or datetime.now(timezone.utc)),
                "retire_reason": reason,
                "certificate_path": cert_dst_rel,
            }
        )
        self._records[factor_id] = retired
        self.save()
        logger.info("registry: retire %s (reason=%s)", factor_id, reason)
        return retired

    # ----------------------------------------------------------------------- read

    def get(self, factor_id: str) -> ProductionFactorRecord:
        rec = self._records.get(factor_id)
        if rec is None:
            raise RegistryError(f"factor_id 不存在：{factor_id!r}")
        return rec

    def __contains__(self, factor_id: str) -> bool:
        return factor_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def list_all(
        self, *, status: ProductionStatus | None = None
    ) -> list[ProductionFactorRecord]:
        """按可选 status 过滤返回全部记录；顺序：(parquet_version, factor_id)。"""
        items = list(self._records.values())
        if status is not None:
            items = [r for r in items if r.status == status]
        return sorted(items, key=lambda r: (r.parquet_version, r.factor_id))

    def list_active(self) -> list[ProductionFactorRecord]:
        return self.list_all(status=ProductionStatus.ACTIVE)

    def list_by_version(
        self, parquet_version: int, *, only_active: bool = True
    ) -> list[ProductionFactorRecord]:
        """按 parquet 版本号过滤。默认只要 active。"""
        if only_active:
            items = self.list_all(status=ProductionStatus.ACTIVE)
        else:
            items = self.list_all()
        return [r for r in items if r.parquet_version == parquet_version]

    @property
    def last_updated_at(self) -> datetime | None:
        ts = self._manifest.get("last_updated_at")
        return datetime.fromisoformat(ts) if ts else None
