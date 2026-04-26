from __future__ import annotations

import unittest

from scripts.lab.log_progress import parse_latest_run_segment, parse_llm_calls_by_loop


class LogProgressTests(unittest.TestCase):
    def test_parse_latest_run_segment_uses_last_loop0_start(self) -> None:
        lines = [
            "2026-01-01 ... Start Loop 0, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
            "Using chat model openai/glm-5.1",
            # 新一次运行再次从 Loop0 开始
            "2026-01-02 ... Start Loop 0, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
        ]
        seg = parse_latest_run_segment(lines)
        self.assertEqual(
            seg,
            [
                "2026-01-02 ... Start Loop 0, Step 0: direct_exp_gen",
                "Using chat model openai/glm-5.1",
            ],
        )

    def test_parse_llm_calls_by_loop_counts_per_loop_in_latest_run(self) -> None:
        lines = [
            # 历史 run（应忽略）
            "Start Loop 0, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
            "Start Loop 1, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
            # 最新 run（应计入）
            "Start Loop 0, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
            "Using chat model openai/glm-5.1",
            "Start Loop 1, Step 0: direct_exp_gen",
            "Using chat model openai/glm-5.1",
            "Start Loop 2, Step 0: direct_exp_gen",
        ]
        stats = parse_llm_calls_by_loop(lines)
        self.assertEqual(stats.total_calls, 3)
        self.assertEqual(stats.calls_by_loop, {0: 2, 1: 1, 2: 0})


if __name__ == "__main__":
    unittest.main()

