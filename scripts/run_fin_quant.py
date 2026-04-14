"""
Run RD-Agent fin_quant with:
- qlib_zhengshi conda for Qlib qrun (see QLIB_RDAGENT_CONDA_ENV / patch_qlib_conda)
- Project YAML templates under rdagent_overrides/ (provider_uri, csi500, SH000905)

Usage (from repo root, conda env qlib_zhengshi activated):

  python scripts/run_fin_quant.py --loop_n=1

Or:

  conda run -n qlib_zhengshi python scripts/run_fin_quant.py --loop_n=1

Requires `.env` with LiteLLM / OpenAI-compatible settings (see `.env.example`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rdagent_integration.patch_qlib_conda import apply_qlib_conda_env_patch

apply_qlib_conda_env_patch()

from rdagent.app.qlib_rd_loop.quant import main

if __name__ == "__main__":
    import fire

    fire.Fire(main)
