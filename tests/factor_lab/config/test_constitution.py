"""Tests: factor_lab.config.constitution byte-parity contract.

The _FALLBACK_CONSTITUTION_TEXT must match the rendered output of
rag_constitution.yaml exactly (byte-for-byte).  If they diverge, the
YAML fallback path silently gives LLM wrong column names / windows.

Run:
    pytest tests/factor_lab/config/test_constitution.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def test_yaml_matches_fallback() -> None:
    """_FALLBACK_CONSTITUTION_TEXT must equal render_constitution(YAML)."""
    from factor_lab.config.constitution import (
        _DEFAULT_YAML_PATH,
        _FALLBACK_CONSTITUTION_TEXT,
        _load_yaml,
        render_constitution,
    )

    cfg = _load_yaml(_DEFAULT_YAML_PATH)
    assert cfg is not None, f"YAML failed to load: {_DEFAULT_YAML_PATH}"

    rendered = render_constitution(cfg)

    if rendered != _FALLBACK_CONSTITUTION_TEXT:
        # Build a human-readable diff to pinpoint where they diverge
        rendered_lines = rendered.splitlines()
        fallback_lines = _FALLBACK_CONSTITUTION_TEXT.splitlines()
        max_lines = max(len(rendered_lines), len(fallback_lines))
        diffs = []
        for i in range(max_lines):
            r = rendered_lines[i] if i < len(rendered_lines) else "<MISSING>"
            f = fallback_lines[i] if i < len(fallback_lines) else "<MISSING>"
            if r != f:
                diffs.append(f"  line {i+1}:\n    YAML:     {r!r}\n    FALLBACK: {f!r}")
            if len(diffs) >= 10:
                diffs.append("  ... (more diffs truncated)")
                break
        raise AssertionError(
            "_FALLBACK_CONSTITUTION_TEXT does not match render_constitution(YAML).\n"
            "Run scripts/lab/_gen_fallback_text.py and update constitution.py.\n"
            "First divergences:\n" + "\n".join(diffs)
        )


def test_fallback_no_forbidden_columns() -> None:
    """Fallback text must not use outdated column names as positive examples.

    $roa2_yearly and SH600000 may appear in a 'do NOT use' warning context — that's fine.
    What we forbid is the old column names being listed as *available* columns.
    """
    from factor_lab.config.constitution import _FALLBACK_CONSTITUTION_TEXT

    # These old column names must NOT appear as available/positive column references.
    # They may still appear in warning notes (e.g. "$roa ... do NOT use them").
    old_positive_columns = ["$rsi12,", "$rsi12 ", "$macd,", "$kdj_k,", "$kdj_k "]
    violations = [col for col in old_positive_columns if col in _FALLBACK_CONSTITUTION_TEXT]
    assert not violations, (
        f"_FALLBACK_CONSTITUTION_TEXT contains old column names as positive refs: {violations}\n"
        "These should be replaced with qfq variants (e.g. $rsi_qfq_12, $macd_qfq, $kdj_k_qfq)."
    )

    # Old instrument format should only appear in "DO NOT" context
    if "SZ000001" in _FALLBACK_CONSTITUTION_TEXT:
        idx = _FALLBACK_CONSTITUTION_TEXT.index("SZ000001")
        context = _FALLBACK_CONSTITUTION_TEXT[max(0, idx - 40):idx + 20]
        assert "NOT" in context or "not" in context, (
            f"SZ000001 appears as positive example: {context!r}"
        )


def test_fallback_correct_windows() -> None:
    """Fallback text must reference the extended window set including 90 and 120."""
    from factor_lab.config.constitution import _FALLBACK_CONSTITUTION_TEXT

    assert "90" in _FALLBACK_CONSTITUTION_TEXT, "Missing window 90 in fallback"
    assert "120" in _FALLBACK_CONSTITUTION_TEXT, "Missing window 120 in fallback"


def test_fallback_correct_instrument_format() -> None:
    """Fallback text must use 000001.SZ format, not SH600000."""
    from factor_lab.config.constitution import _FALLBACK_CONSTITUTION_TEXT

    assert "000001.SZ" in _FALLBACK_CONSTITUTION_TEXT, (
        "Fallback missing correct instrument format '000001.SZ'"
    )
    # Allow 'SH600000' only in a "DO NOT" context — check it's not a positive example
    # The text should say "DO NOT rewrite to 'SH600000'"
    if "SH600000" in _FALLBACK_CONSTITUTION_TEXT:
        idx = _FALLBACK_CONSTITUTION_TEXT.index("SH600000")
        context = _FALLBACK_CONSTITUTION_TEXT[max(0, idx-30):idx+20]
        assert "NOT" in context or "not" in context, (
            f"SH600000 appears in fallback without 'NOT' context: {context!r}"
        )
