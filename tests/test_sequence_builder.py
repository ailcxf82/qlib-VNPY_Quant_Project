import unittest
import numpy as np
import pandas as pd

from datasets.sequence_builder import build_panel_sequences


class TestSequenceBuilder(unittest.TestCase):
    def test_last_day_equals_sample_day_and_no_future(self):
        # 使用工作日作为“交易日”日历（单测不依赖 qlib）
        cal = pd.bdate_range("2025-01-01", periods=80).normalize()
        dts = cal[:70]

        idx = pd.MultiIndex.from_product([dts, ["AAA"]], names=["datetime", "instrument"])
        X = pd.DataFrame(np.random.randn(len(idx), 5), index=idx, columns=[f"f{i}" for i in range(5)])

        res = build_panel_sequences(
            X,
            seq_len=60,
            calendar=cal,
            require_consecutive_trading_days=True,
            check=True,  # 这里会触发内部断言
        )
        self.assertEqual(res.X.shape[1], 60)
        self.assertEqual(res.X.shape[2], 5)
        # AAA 有 70 个交易日数据，满窗序列数 = 70 - 60 + 1 = 11
        self.assertEqual(res.X.shape[0], 11)

        # 再额外验证一个样本：最后一天等于该样本当天特征（显式检查）
        end = res.endpoint_index[0]
        np.testing.assert_allclose(res.X[0, -1, :], X.loc[end].values, rtol=1e-6, atol=1e-6)

    def test_missing_day_breaks_window_when_strict(self):
        cal = pd.bdate_range("2025-01-01", periods=80).normalize()
        dts = cal[:70].delete(10)  # 人为删掉一个交易日（模拟停牌/缺失）
        idx = pd.MultiIndex.from_product([dts, ["BBB"]], names=["datetime", "instrument"])
        X = pd.DataFrame(np.random.randn(len(idx), 3), index=idx, columns=["a", "b", "c"])

        res = build_panel_sequences(
            X,
            seq_len=20,
            calendar=cal,
            require_consecutive_trading_days=True,
            check=True,
        )
        # 因为缺了一个交易日，跨越缺口的窗口都会被丢弃，所以有效序列数会小于满窗的 70-20+1
        self.assertLess(res.X.shape[0], (len(dts) - 20 + 1))


if __name__ == "__main__":
    unittest.main()



