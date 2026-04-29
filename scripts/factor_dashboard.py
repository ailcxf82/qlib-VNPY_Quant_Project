"""
因子管理仪表板 — Factor Dashboard
====================================
显示 RD-Agent 产出的所有因子实验，支持一键应用到 Qlib 生产配置。

启动方式：
    conda activate qlib_zhengshi
    streamlit run scripts/factor_dashboard.py

或直接：
    python -m streamlit run scripts/factor_dashboard.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ─── 路径常量 ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WS_BASE = ROOT / "git_ignore_folder" / "RD-Agent_workspace"
PROD_PARQUET = ROOT / "git_ignore_folder" / "combined_factors_df.parquet"
CONF_COMBINED = ROOT / "rdagent_overrides" / "factor_template" / "conf_combined_factors.yaml"
FACTOR_LAB_CFG = ROOT / "config" / "factor_lab.yaml"
REGISTRY_MANIFEST = ROOT / "factor_registry" / "data" / "manifest.json"
REGISTRY_PARQUET_DIR = ROOT / "factor_registry" / "parquet"

# ─── 页面配置 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="因子管理仪表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-good  { color: #00c060; font-weight: bold; }
    .metric-bad   { color: #e04040; font-weight: bold; }
    .metric-mid   { color: #e0a000; font-weight: bold; }
    .tag          { background:#2b3a4a; border-radius:4px; padding:2px 6px;
                    font-size:0.82em; margin:2px; display:inline-block; }
    .applied-box  { background:#1a3a1a; border:1px solid #00c060;
                    border-radius:6px; padding:12px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)


# ─── 数据加载 ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="扫描实验工作区…")
def load_experiments() -> pd.DataFrame:
    rows = []
    for ws_dir in WS_BASE.iterdir():
        if not ws_dir.is_dir():
            continue
        csv_path = ws_dir / "qlib_res.csv"
        pq_path = ws_dir / "combined_factors_df.parquet"
        if not csv_path.exists():
            continue
        try:
            df_csv = pd.read_csv(csv_path, index_col=0)
            df_csv.columns = ["value"]
            m = df_csv["value"].to_dict()
        except Exception:
            continue

        icir = m.get("ICIR")
        ic = m.get("IC")
        rank_ic = m.get("Rank IC")
        rank_icir = m.get("Rank ICIR")
        composite = m.get("composite_score") or m.get("1day.composite_score")
        l2_train = m.get("l2.train")
        l2_valid = m.get("l2.valid")

        if icir is None or pd.isna(float(icir)):
            continue

        # Factor names
        factors: list[str] = []
        if pq_path.exists():
            try:
                df_pq = pd.read_parquet(pq_path)
                lvl = df_pq.columns.get_level_values(-1) if df_pq.columns.nlevels > 1 else df_pq.columns
                factors = list(lvl)
            except Exception:
                pass

        # Date config from conf_baseline.yaml
        conf = ws_dir / "conf_baseline.yaml"
        test_start = test_end = train_start = train_end = ""
        if conf.exists():
            text = conf.read_text(encoding="utf-8-sig")
            import re
            # backtest start/end
            m_bt = re.search(r"backtest:\s*\n\s+start_time:\s*(\S+)\s*\n\s+end_time:\s*(\S+)", text)
            if m_bt:
                test_start, test_end = m_bt.group(1), m_bt.group(2)
            # train segments
            m_tr = re.search(r"train:\s*\[([^\]]+)\]", text)
            if m_tr:
                parts = [x.strip() for x in m_tr.group(1).split(",")]
                if len(parts) == 2:
                    train_start, train_end = parts

        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        rows.append({
            "workspace_id": ws_dir.name,
            "run_date": mtime,
            "IC": round(float(ic), 4) if ic else None,
            "ICIR": round(float(icir), 3) if icir else None,
            "Rank_IC": round(float(rank_ic), 4) if rank_ic else None,
            "Rank_ICIR": round(float(rank_icir), 3) if rank_icir else None,
            "composite": round(float(composite), 2) if composite else None,
            "l2_train": round(float(l2_train), 4) if l2_train else None,
            "l2_valid": round(float(l2_valid), 4) if l2_valid else None,
            "factors": factors,
            "n_factors": len(factors),
            "test_period": f"{test_start} → {test_end}" if test_start else "unknown",
            "train_period": f"{train_start} → {train_end}" if train_start else "unknown",
            "has_parquet": pq_path.exists(),
            "parquet_path": str(pq_path) if pq_path.exists() else "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("ICIR", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


@st.cache_data(ttl=30)
def load_production_factors() -> dict:
    # 优先使用 L3 registry（当前 production 真实口径）
    if REGISTRY_MANIFEST.exists() and FACTOR_LAB_CFG.exists():
        try:
            mf = json.loads(REGISTRY_MANIFEST.read_text(encoding="utf-8"))
            cfg = yaml.safe_load(FACTOR_LAB_CFG.read_text(encoding="utf-8")) or {}
            cur_ver = int((cfg.get("registry") or {}).get("current_parquet_version", 0))
            active = [
                r for r in mf.get("factors", [])
                if isinstance(r, dict)
                and r.get("status") == "active"
                and int(r.get("parquet_version", -1)) == cur_ver
            ]
            factors = [str(r.get("parquet_column") or r.get("name") or r.get("factor_name")) for r in active]
            date_min = None
            date_max = None
            if active:
                mins = [r.get("date_min") for r in active if r.get("date_min")]
                maxs = [r.get("date_max") for r in active if r.get("date_max")]
                date_min = min(mins) if mins else None
                date_max = max(maxs) if maxs else None
            pq = REGISTRY_PARQUET_DIR / f"factors_v{cur_ver}.parquet"
            shape = None
            try:
                if pq.exists():
                    df = pd.read_parquet(pq, columns=factors or None)
                    shape = df.shape
            except Exception:
                # 可能缺 pyarrow，忽略 shape，仅展示 manifest 信息
                pass
            return {
                "factors": factors,
                "shape": shape,
                "date_range": f"{date_min} → {date_max}" if date_min and date_max else "N/A",
                "source": "registry",
                "version": cur_ver,
                "workspace_ids": sorted(set(str(r.get("workspace_id", "")) for r in active if r.get("workspace_id"))),
                "parquet_path": str(pq),
            }
        except Exception:
            pass

    # fallback：旧 combined_factors_df.parquet 口径
    if not PROD_PARQUET.exists():
        return {"factors": [], "shape": None, "date_range": "N/A", "source": "legacy"}
    try:
        df = pd.read_parquet(PROD_PARQUET)
        lvl = df.columns.get_level_values(-1) if df.columns.nlevels > 1 else df.columns
        factors = list(lvl)
        dates = df.index.get_level_values("datetime")
        return {
            "factors": factors,
            "shape": df.shape,
            "date_range": f"{dates.min().date()} → {dates.max().date()}",
            "source": "legacy",
            "version": None,
            "workspace_ids": [],
            "parquet_path": str(PROD_PARQUET),
        }
    except Exception as e:
        return {"factors": [], "shape": None, "date_range": f"错误: {e}", "source": "legacy"}


# ─── 帮助函数 ───────────────────────────────────────────────────────────────
def _icir_color(v: float) -> str:
    if v >= 5:
        return "metric-good"
    if v >= 2:
        return "metric-mid"
    return "metric-bad"


def _icir_badge(v: float | None) -> str:
    if v is None:
        return "—"
    cls = _icir_color(v)
    return f'<span class="{cls}">{v:.3f}</span>'


def generate_qlib_yaml_snippet(factors: list[str], workspace_id: str) -> str:
    """生成包含所选因子的 qlib StaticDataLoader 配置片段（供复制使用）。"""
    factor_block = "\n".join(f"        - {f}" for f in factors)
    return f"""# ── 由因子仪表板生成 ─────────────────────────────────
# workspace: {workspace_id}
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# 因子列表 ({len(factors)} 个):
{factor_block}

# StaticDataLoader 配置段（粘贴到 conf_combined_factors.yaml 中的 dataloader_l）：
- class: qlib.data.dataset.loader.StaticDataLoader
  kwargs:
    config: "combined_factors_df.parquet"
    # 该 parquet 文件须包含以下列（已是 ('feature', col) 双索引格式）：
    # {factors}
"""


def apply_factors_to_production(workspace_id: str, parquet_path: str) -> tuple[bool, str]:
    """将选中 workspace 的 parquet 拷贝到生产目录并备份旧文件。"""
    src = Path(parquet_path)
    if not src.exists():
        return False, f"源文件不存在: {src}"
    try:
        # 备份旧文件
        if PROD_PARQUET.exists():
            bak = PROD_PARQUET.with_suffix(f".parquet.bak.{int(datetime.now().timestamp())}")
            shutil.copy2(PROD_PARQUET, bak)
        shutil.copy2(src, PROD_PARQUET)
        return True, f"已成功应用！备份旧文件于同目录 *.bak.*"
    except Exception as e:
        return False, str(e)


# ─── 主界面 ──────────────────────────────────────────────────────────────────
st.title("📈 因子管理仪表板")
st.caption("扫描 RD-Agent 实验结果，一键浏览、对比与应用到 Qlib 生产配置")

# 侧边栏：筛选控件
with st.sidebar:
    st.header("筛选 & 显示")
    min_icir = st.slider("最低 ICIR 阈值", 0.0, 6.0, 0.0, 0.1)
    min_n_factors = st.slider("最少因子数", 0, 20, 0, 1)
    max_n_factors = st.slider("最多因子数", 1, 30, 30, 1)
    only_has_parquet = st.checkbox("仅显示有 parquet 的实验", value=True)
    test_period_filter = st.selectbox(
        "测试期过滤",
        ["全部", "2026 年测试期", "2025 年测试期", "2024 及更早"],
    )
    st.divider()
    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_explore, tab_detail, tab_production = st.tabs(["🔍 因子实验列表", "📋 详情 & 应用", "🏭 生产状态"])

# ── Tab 1: 实验列表 ──────────────────────────────────────────────────────────
with tab_explore:
    df = load_experiments()

    if df.empty:
        st.warning("未找到任何带指标的因子实验，请确认 RD-Agent workspace 路径正确。")
        st.stop()

    # 应用筛选
    mask = (df["ICIR"] >= min_icir) & (df["n_factors"] >= min_n_factors) & (df["n_factors"] <= max_n_factors)
    if only_has_parquet:
        mask &= df["has_parquet"]
    if test_period_filter == "2026 年测试期":
        mask &= df["test_period"].str.contains("2026")
    elif test_period_filter == "2025 年测试期":
        mask &= df["test_period"].str.contains("2025") & ~df["test_period"].str.contains("2026")
    elif test_period_filter == "2024 及更早":
        mask &= ~df["test_period"].str.contains("202[56]")
    df_show = df[mask].copy()

    # 统计摘要
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("实验总数", len(df))
    col2.metric("筛选后数量", len(df_show))
    col3.metric("最高 ICIR", f"{df_show['ICIR'].max():.3f}" if not df_show.empty else "—")
    col4.metric("中位 ICIR", f"{df_show['ICIR'].median():.3f}" if not df_show.empty else "—")
    col5.metric("ICIR>5 实验数", int((df_show["ICIR"] >= 5).sum()))

    st.divider()

    if df_show.empty:
        st.info("当前筛选条件下无实验，请调整左侧滑块。")
    else:
        # 构建展示表格
        disp = df_show[[
            "rank", "ICIR", "IC", "Rank_ICIR", "Rank_IC",
            "composite", "n_factors", "test_period", "run_date",
        ]].copy()
        disp["run_date"] = pd.to_datetime(disp["run_date"]).dt.strftime("%m/%d %H:%M")
        disp.columns = ["排名", "ICIR", "IC", "Rank ICIR", "Rank IC", "综合分", "因子数", "测试期", "运行时间"]

        # 颜色标注 ICIR 列
        def color_icir(val):
            if val >= 5:
                return "background-color: #0a2a0a; color: #00e070"
            if val >= 2:
                return "background-color: #2a1a00; color: #e0a000"
            return "background-color: #2a0a0a; color: #e04040"

        styled = (
            disp.style
            .applymap(color_icir, subset=["ICIR"])
            .format({
                "ICIR": "{:.3f}",
                "IC": lambda x: f"{x:.4f}" if pd.notna(x) else "—",
                "Rank ICIR": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                "Rank IC": lambda x: f"{x:.4f}" if pd.notna(x) else "—",
                "综合分": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
        )

        st.dataframe(styled, use_container_width=True, height=500)

        # 点击行 → 跳转到详情
        st.info("在「详情 & 应用」标签页中，选择一个实验的 Workspace ID 来查看详细信息并应用配置。")

        # 快速选择区
        st.subheader("快速选择")
        preset_options = {
            f"#{row['rank']} ICIR={row['ICIR']:.3f}  {row['factors'][:3]}{'…' if len(row['factors'])>3 else ''}  [{row['test_period']}]": row["workspace_id"]
            for _, row in df_show.head(30).iterrows()
        }
        chosen_label = st.selectbox("选择实验（按排名排序）", list(preset_options.keys()), key="quick_select")
        chosen_ws_id = preset_options[chosen_label]

        if st.button("📋 查看详情 & 应用此实验", type="primary", use_container_width=True, key="go_detail"):
            st.session_state["selected_ws"] = chosen_ws_id
            st.info(f"已选中 `{chosen_ws_id}`，请切换到「详情 & 应用」标签页。")


# ── Tab 2: 详情 & 应用 ───────────────────────────────────────────────────────
with tab_detail:
    df = load_experiments()

    # 选择 workspace
    all_ws_ids = df["workspace_id"].tolist() if not df.empty else []
    default_idx = 0
    if "selected_ws" in st.session_state and st.session_state["selected_ws"] in all_ws_ids:
        default_idx = all_ws_ids.index(st.session_state["selected_ws"])

    selected_ws = st.selectbox(
        "选择实验 Workspace ID",
        all_ws_ids,
        index=default_idx,
        key="detail_select",
        help="可直接在列表页点击「查看详情 & 应用」快速跳转",
    )

    if not selected_ws or df.empty:
        st.info("请先在「因子实验列表」中选择一个实验。")
        st.stop()

    row = df[df["workspace_id"] == selected_ws].iloc[0]

    # 指标展示
    st.subheader(f"📊 实验指标 — 排名 #{int(row['rank'])}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ICIR", f"{row['ICIR']:.3f}")
    c2.metric("IC", f"{row['IC']:.4f}" if pd.notna(row["IC"]) else "—")
    c3.metric("Rank ICIR", f"{row['Rank_ICIR']:.3f}" if pd.notna(row["Rank_ICIR"]) else "—")
    c4.metric("Rank IC", f"{row['Rank_IC']:.4f}" if pd.notna(row["Rank_IC"]) else "—")
    c5.metric("综合分", f"{row['composite']:.2f}" if pd.notna(row["composite"]) else "—")
    c6.metric("因子数量", row["n_factors"])

    col_a, col_b = st.columns(2)
    col_a.info(f"**测试期（回测）：** {row['test_period']}")
    col_b.info(f"**训练期：** {row['train_period']}")

    st.caption(f"运行时间: {row['run_date'].strftime('%Y-%m-%d %H:%M')}  |  workspace: `{selected_ws}`")

    st.divider()

    # 因子列表
    st.subheader("🧩 因子列表")
    factors = row["factors"]
    if not factors:
        st.warning("该实验没有 combined_factors_df.parquet，无法读取自定义因子名称。仅使用基础特征运行。")
    else:
        cols = st.columns(min(4, len(factors)))
        for i, f in enumerate(factors):
            cols[i % len(cols)].markdown(
                f'<span class="tag">#{i+1} {f}</span>', unsafe_allow_html=True
            )
        st.caption(f"共 {len(factors)} 个 AI 生成因子，配合 34 个基础特征（价量/估值/技术）一同送入 GBDT 训练。")

    st.divider()

    # Qlib 配置片段
    st.subheader("📄 Qlib 配置应用方式")

    apply_mode = st.radio(
        "应用方式",
        ["① 直接替换生产 parquet（推荐）", "② 仅查看/复制 YAML 片段", "③ 手动合并到指定 YAML 文件"],
        horizontal=True,
    )

    if apply_mode == "① 直接替换生产 parquet（推荐）":
        st.markdown("""
        **此操作将：**
        1. 备份当前 `git_ignore_folder/combined_factors_df.parquet`（自动加 `.bak.时间戳` 后缀）
        2. 将所选实验的因子数据（含列名）复制为新的生产 parquet
        3. `conf_combined_factors.yaml` 无需修改（已配置为相对路径读取）
        """)

        if not row["has_parquet"]:
            st.error("该实验无 parquet 文件，无法应用。")
        else:
            with st.expander("查看所选 parquet 路径"):
                st.code(row["parquet_path"])

            apply_confirmed = st.checkbox("✅ 我已确认，要将此因子集应用到生产环境")
            if apply_confirmed:
                if st.button("🚀 立即应用到生产", type="primary", use_container_width=True):
                    ok, msg = apply_factors_to_production(selected_ws, row["parquet_path"])
                    if ok:
                        st.success(f"✅ 应用成功！{msg}")
                        st.markdown(f"""
<div class="applied-box">
已应用因子组合（{len(factors)} 个因子）：<br>
{'  '.join(f'<span class="tag">{f}</span>' for f in factors)}<br><br>
<b>后续步骤：</b>重新运行 Qlib 回测即可使用新因子。
</div>
""", unsafe_allow_html=True)
                        load_production_factors.clear()
                    else:
                        st.error(f"❌ 应用失败：{msg}")

    elif apply_mode == "② 仅查看/复制 YAML 片段":
        snippet = generate_qlib_yaml_snippet(factors, selected_ws)
        st.code(snippet, language="yaml")
        st.caption("将上方 StaticDataLoader 片段粘贴到 rdagent_overrides/factor_template/conf_combined_factors.yaml 的 dataloader_l 列表中。")

    else:  # 手动合并
        st.info("请在下方指定目标 YAML 文件路径，系统将自动替换其 `config: \"combined_factors_df.parquet\"` 所在的 StaticDataLoader 块。")
        target_yaml = st.text_input(
            "目标 YAML 路径",
            value=str(CONF_COMBINED),
        )
        if st.button("打开文件（在 Cursor IDE 中）") and Path(target_yaml).exists():
            st.code(Path(target_yaml).read_text(encoding="utf-8-sig"), language="yaml")

    st.divider()

    # 与其他实验对比
    st.subheader("📊 与相邻排名对比")
    rank_current = int(row["rank"])
    compare_range = df[df["rank"].between(max(1, rank_current - 2), rank_current + 2)].copy()
    compare_disp = compare_range[["rank", "ICIR", "IC", "Rank_ICIR", "composite", "n_factors", "test_period"]].copy()
    compare_disp.columns = ["排名", "ICIR", "IC", "Rank ICIR", "综合分", "因子数", "测试期"]

    def highlight_current(row_):
        return ["background-color: #1a3a2a" if row_["排名"] == rank_current else "" for _ in row_]

    st.dataframe(
        compare_disp.style.apply(highlight_current, axis=1).format({
            "ICIR": "{:.3f}", "IC": "{:.4f}", "Rank ICIR": "{:.3f}",
            "综合分": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 3: 生产状态 ──────────────────────────────────────────────────────────
with tab_production:
    prod = load_production_factors()
    df_all = load_experiments()

    st.subheader("🏭 当前生产因子状态")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("数据来源", "L3 Registry" if prod.get("source") == "registry" else "Legacy Parquet")
        if prod.get("version"):
            st.metric("生产版本", f"v{prod['version']}")
        st.metric("生产因子数量", len(prod["factors"]))
        st.metric("数据日期范围", prod["date_range"])
        if prod["shape"]:
            st.metric("数据条数", f"{prod['shape'][0]:,}")
    with col2:
        st.markdown("**当前生产因子列表：**")
        if prod["factors"]:
            for f in prod["factors"]:
                st.markdown(f'<span class="tag">{f}</span>', unsafe_allow_html=True)
        else:
            st.warning("无法读取生产因子。")
        if prod.get("workspace_ids"):
            st.caption(f"workspace_ids: {', '.join(prod['workspace_ids'])}")
        if prod.get("parquet_path"):
            st.caption(f"parquet: {prod['parquet_path']}")

    st.divider()
    st.subheader("📈 TOP-10 最优实验（可直接应用）")

    if not df_all.empty:
        top10 = df_all[df_all["has_parquet"]].head(10)
        for _, r in top10.iterrows():
            with st.expander(
                f"#{int(r['rank'])}  ICIR={r['ICIR']:.3f}  IC={r['IC']:.4f}  "
                f"[{len(r['factors'])} 因子]  测试期: {r['test_period']}"
            ):
                st.markdown("**因子列表：**")
                for f in r["factors"]:
                    st.markdown(f'<span class="tag">{f}</span>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                col_a.metric("ICIR", f"{r['ICIR']:.3f}")
                col_a.metric("Rank ICIR", f"{r['Rank_ICIR']:.3f}" if pd.notna(r["Rank_ICIR"]) else "—")
                col_b.metric("IC", f"{r['IC']:.4f}" if pd.notna(r["IC"]) else "—")
                col_b.metric("综合分", f"{r['composite']:.2f}" if pd.notna(r["composite"]) else "—")
                st.caption(f"workspace: `{r['workspace_id']}` | 运行于 {r['run_date'].strftime('%Y-%m-%d %H:%M')}")

                if st.button(f"🚀 应用 #{int(r['rank'])} 到生产", key=f"apply_{r['workspace_id']}"):
                    st.session_state["selected_ws"] = r["workspace_id"]
                    ok, msg = apply_factors_to_production(r["workspace_id"], r["parquet_path"])
                    if ok:
                        st.success(f"✅ 已应用！{msg}")
                        load_production_factors.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    st.divider()
    st.subheader("📦 备份历史")
    bak_files = sorted(PROD_PARQUET.parent.glob("combined_factors_df.parquet.bak.*"), reverse=True)
    if bak_files:
        for b in bak_files[:5]:
            mtime = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = b.stat().st_size / 1024 / 1024
            st.text(f"  {b.name}  ({size_mb:.1f} MB)  {mtime}")
    else:
        st.info("暂无备份文件。")

    st.divider()
    st.subheader("⚙️ Qlib 配置快速参考")
    st.markdown("""
**使用自定义因子的 `conf_combined_factors.yaml` 关键段：**

```yaml
# StaticDataLoader：从 workspace 本地 parquet 加载 AI 因子
- class: qlib.data.dataset.loader.StaticDataLoader
  kwargs:
    config: "combined_factors_df.parquet"   # 相对路径，qrun 以 workspace 为 CWD
```

**运行完整因子回测：**
```bash
# 将所选实验的 parquet 复制到生产目录后
cd D:\\quant_project\\Qlib_Quant\\qlib-VNPY_Quant_Project
conda activate qlib_zhengshi
python scripts/lab/run_rdagent_loop.py --mode factor --loop_n 10
```

**仅做单次 qrun 验证（在 WSL 中）：**
```bash
cd /path/to/workspace && qrun conf_combined_factors.yaml
```
""")
