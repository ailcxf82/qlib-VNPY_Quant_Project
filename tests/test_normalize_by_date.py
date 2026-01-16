import unittest
import numpy as np
import pandas as pd

from utils.normalize import normalize_by_date


class TestNormalizeByDate(unittest.TestCase):
    def test_demean_and_zscore(self):
        data = {
            "date": ["2025-01-01"] * 3 + ["2025-01-02"] * 2,
            "code": ["A", "B", "C", "A", "B"],
            "pred_lgb": [1.0, 2.0, 3.0, 10.0, 20.0],
            "pred_gru": [2.0, 4.0, 6.0, 20.0, 40.0],
        }
        df = pd.DataFrame(data)

        # demean
        df_demean = normalize_by_date(df, ["pred_lgb", "pred_gru"], mode="demean")
        for d, g in df_demean.groupby("date"):
            self.assertAlmostEqual(g["pred_lgb"].mean(), 0.0, places=6)
            self.assertAlmostEqual(g["pred_gru"].mean(), 0.0, places=6)

        # zscore
        df_z = normalize_by_date(df, ["pred_lgb", "pred_gru"], mode="zscore", eps=1e-6)
        for d, g in df_z.groupby("date"):
            self.assertAlmostEqual(g["pred_lgb"].mean(), 0.0, places=6)
            self.assertAlmostEqual(g["pred_gru"].mean(), 0.0, places=6)
            self.assertAlmostEqual(g["pred_lgb"].std(), 1.0, places=3)
            self.assertAlmostEqual(g["pred_gru"].std(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()



