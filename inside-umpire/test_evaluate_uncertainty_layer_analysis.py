import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from pipeline import evaluate_uncertainty as eval_unc


class EvaluateUncertaintyLayerAnalysisTests(unittest.TestCase):
    def test_sort_layer_strategies_prefers_numeric_layers_then_special_layers(self):
        layers = ["mean_pooling", "layer_12", "last_layer", "layer_3", "layer_0", "layer_21"]

        sorted_layers = eval_unc.sort_layer_strategies(layers)

        self.assertEqual(
            sorted_layers,
            ["layer_0", "layer_3", "layer_12", "layer_21", "last_layer", "mean_pooling"],
        )

    def test_build_layer_metric_table_expands_metrics_per_signal_and_layer(self):
        layered_results = {
            "warning_signal": {
                "layer_0": {"auc": 0.61, "cece": 0.12},
                "layer_12": {"auc": 0.73, "cece": 0.08},
            },
            "confidence_signal": {
                "layer_0": {"auc": 0.55, "cece": 0.15},
            },
        }

        table = eval_unc.build_layer_metric_table(layered_results)

        self.assertEqual(
            list(table.columns),
            ["signal", "layer", "layer_order", "auc", "cece"],
        )
        self.assertEqual(len(table), 3)
        self.assertEqual(table.iloc[0]["signal"], "confidence_signal")
        self.assertEqual(table.iloc[0]["layer"], "layer_0")
        self.assertEqual(table.iloc[1]["signal"], "warning_signal")
        self.assertEqual(table.iloc[1]["layer"], "layer_0")
        self.assertEqual(table.iloc[2]["layer"], "layer_12")

    def test_select_layer_strategies_for_analysis_keeps_representative_layers_by_default(self):
        layers = ["layer_0", "layer_1", "layer_3", "layer_6", "layer_12", "layer_21", "last_layer", "mean_pooling"]

        selected_layers = eval_unc.select_layer_strategies_for_analysis(layers)

        self.assertEqual(
            selected_layers,
            ["layer_0", "layer_3", "layer_6", "layer_12", "layer_21", "last_layer", "mean_pooling"],
        )

    def test_evaluate_layer_signals_uses_passed_layer_subset(self):
        image_df = pd.DataFrame(
            {
                "uncertainty_layer_mean_0": [0.1, 0.9],
                "uncertainty_layer_mean_3": [0.2, 0.8],
            }
        )

        with mock.patch.object(eval_unc, "evaluate_scores_against_labels", return_value={"auc": 0.6}):
            layered_results = eval_unc.evaluate_layer_signals(
                image_df=image_df,
                labels=[0, 1],
                eval_col="correct",
                layer_strategies=["layer_0"],
                layer_feature_types=["mean"],
            )

        self.assertEqual(list(layered_results.keys()), ["layer_mean"])
        self.assertEqual(list(layered_results["layer_mean"].keys()), ["layer_0"])

    def test_plot_layer_metric_trends_creates_png_per_signal(self):
        table = pd.DataFrame(
            [
                {"signal": "warning_signal", "layer": "layer_0", "layer_order": 0, "auc": 0.61},
                {"signal": "warning_signal", "layer": "layer_12", "layer_order": 12, "auc": 0.73},
                {"signal": "confidence_signal", "layer": "layer_0", "layer_order": 0, "auc": 0.55},
                {"signal": "confidence_signal", "layer": "last_layer", "layer_order": 10_000, "auc": 0.69},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_unc.plot_layer_metric_trends(table, tmpdir, metric_name="auc")

            warning_path = os.path.join(tmpdir, "warning_signal_auc_by_layer.png")
            confidence_path = os.path.join(tmpdir, "confidence_signal_auc_by_layer.png")
            combined_path = os.path.join(tmpdir, "all_layer_signals_auc_by_layer.png")

            self.assertTrue(os.path.exists(warning_path))
            self.assertTrue(os.path.exists(confidence_path))
            self.assertTrue(os.path.exists(combined_path))


if __name__ == "__main__":
    unittest.main()
