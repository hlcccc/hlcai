import argparse
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(".")

from modules.eval_utils import (
    compute_aurac_from_image_df,
    compute_pearsonr,
    df_to_markdown_bold,
    get_calibrate_ece,
    get_tpr_at_fpr,
)
from modules.logdet_utils import compute_logdet, compute_probL1_logdet, get_normL1_prob, normalize_embedding


def load_pickle_with_numpy_compat(file_path):
    try:
        with open(file_path, "rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        import numpy as np

        sys.modules.setdefault("numpy._core", np.core)
        with open(file_path, "rb") as handle:
            return pickle.load(handle)


parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--jitter", type=float, default=1e-8)
parser.add_argument("--uncertainty_weight", type=float, default=0.5)
parser.add_argument("--layer_plot_metric", type=str, default="auc")
parser.add_argument("--layer_analysis_scope", type=str, default="representative", choices=["representative", "all"])


REPRESENTATIVE_LAYER_STRATEGIES = [
    "layer_0",
    "layer_3",
    "layer_6",
    "layer_9",
    "layer_12",
    "layer_15",
    "layer_18",
    "layer_21",
    "last_layer",
    "mean_pooling",
]


def safe_mean(values, default=0.0):
    if values is None:
        return default
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return default
    return float(np.mean(values))


def compute_generation_diversity(generations_text):
    if not generations_text or len(generations_text) < 2:
        return 0.0
    return float(len(set(generations_text)) / max(len(generations_text), 1))


def get_signal_summary(uncertainty_info):
    if not uncertainty_info:
        return {}
    return uncertainty_info.get("signal_summary", {}) or {}


def get_layer_feature_by_strategy(uncertainty_info, strategy, feature_name):
    if not uncertainty_info or "layer_features_by_strategy" not in uncertainty_info:
        return 0.0
    features = uncertainty_info["layer_features_by_strategy"].get(strategy, {})
    return float(features.get(feature_name, 0.0))


def get_row_signal(row, signal_name):
    uncertainty_list = row.get("uncertainty_info", [])
    if not uncertainty_list:
        if signal_name == "generation_diversity":
            return compute_generation_diversity(row.get("generations_text", []))
        return 0.0

    values = []
    for uncertainty_info in uncertainty_list:
        summary = get_signal_summary(uncertainty_info)
        if signal_name == "entropy_signal":
            values.append(summary.get("entropy_signal", safe_mean(uncertainty_info.get("token_entropies", []))))
        elif signal_name == "confidence_signal":
            if "confidence_signal" in summary:
                values.append(summary["confidence_signal"])
            else:
                values.append(1.0 - safe_mean(uncertainty_info.get("confidence_scores", []), default=1.0))
        elif signal_name == "warning_signal":
            values.append(summary.get("warning_signal", safe_mean(uncertainty_info.get("warning_scores", []))))
        elif signal_name == "layer_instability":
            if "layer_instability" in summary:
                values.append(summary["layer_instability"])
            else:
                spread = safe_mean(uncertainty_info.get("layer_spreads", []))
                drift = safe_mean(uncertainty_info.get("temporal_drifts", []))
                values.append(0.5 * spread + 0.5 * drift)
        elif signal_name == "early_stop_rate":
            values.append(1.0 if uncertainty_info.get("early_stop_triggered", False) else 0.0)
        elif signal_name == "generation_diversity":
            values.append(compute_generation_diversity(row.get("generations_text", [])))
        elif signal_name == "online_uncertainty":
            values.append(float(uncertainty_info.get("combined_online_uncertainty", 0.0)))
        elif signal_name.startswith("layer_"):
            parts = signal_name.split("_")
            feature_name = parts[1]
            strategy = "_".join(parts[2:])
            values.append(get_layer_feature_by_strategy(uncertainty_info, strategy, feature_name))
    return safe_mean(values)


def discover_layer_strategies(df):
    strategies = set()
    if "uncertainty_info" not in df.columns:
        return []
    for uncertainty_list in df["uncertainty_info"]:
        if not isinstance(uncertainty_list, list):
            continue
        for uncertainty_info in uncertainty_list:
            strategies.update(uncertainty_info.get("layer_features_by_strategy", {}).keys())
        if strategies:
            break
    return sort_layer_strategies(strategies)


def get_layer_order(strategy):
    if strategy == "last_layer":
        return 10_000
    if strategy == "mean_pooling":
        return 20_000
    match = re.fullmatch(r"layer_(\d+)", strategy)
    if match:
        return int(match.group(1))
    return 30_000


def sort_layer_strategies(strategies):
    return sorted(strategies, key=lambda strategy: (get_layer_order(strategy), strategy))


def select_layer_strategies_for_analysis(layer_strategies, scope="representative"):
    sorted_layers = sort_layer_strategies(layer_strategies)
    if scope == "all":
        return sorted_layers
    representative_layers = [layer for layer in REPRESENTATIVE_LAYER_STRATEGIES if layer in sorted_layers]
    return representative_layers if representative_layers else sorted_layers


def evaluate_scores_against_labels(scores, labels, eval_col):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    valid_idx = np.isfinite(scores)
    if np.sum(valid_idx) < 2 or len(np.unique(labels[valid_idx])) < 2:
        return None

    valid_scores = scores[valid_idx]
    valid_labels = labels[valid_idx]
    method_df = pd.DataFrame({"score": valid_scores, eval_col: valid_labels})
    wrong_scores = valid_scores[valid_labels == 0]
    correct_scores = valid_scores[valid_labels == 1]
    if len(wrong_scores) == 0 or len(correct_scores) == 0:
        return None

    try:
        auc = roc_auc_score(1 - valid_labels, valid_scores)
        cece = get_calibrate_ece(method_df, "score", eval_col=eval_col, num_bins=15, model_type="minmax")
        pearsonr = np.abs(compute_pearsonr(valid_scores, valid_labels, num_bins=50)[0])
        tpr_at_01 = get_tpr_at_fpr(correct_scores, wrong_scores, fpr_threshold=0.1)
        tpr_at_001 = get_tpr_at_fpr(correct_scores, wrong_scores, fpr_threshold=0.01)
        aurac = compute_aurac_from_image_df(method_df, "score", uncertainty=True, eval_col=eval_col)
        return {
            "auc": auc,
            "cece": cece,
            "pearsonr": pearsonr,
            "tpr_at_0.1_fpr": tpr_at_01,
            "tpr_at_0.01_fpr": tpr_at_001,
            "aurac": aurac,
        }
    except Exception as exc:
        print(f"Error evaluating layer scores: {exc}")
        return None


def evaluate_layer_signals(image_df, labels, eval_col, layer_strategies, layer_feature_types):
    layered_results = {}
    for feature_type in layer_feature_types:
        signal_name = f"layer_{feature_type}"
        per_layer_results = {}
        for strategy in layer_strategies:
            strategy_name = strategy.replace("%", "pct").replace("layer_", "")
            col_name = f"uncertainty_layer_{feature_type}_{strategy_name}"
            if col_name not in image_df.columns:
                continue
            result = evaluate_scores_against_labels(image_df[col_name].values, labels, eval_col)
            if result is not None:
                per_layer_results[strategy] = result
        if per_layer_results:
            layered_results[signal_name] = per_layer_results
    return layered_results


def build_layer_metric_table(layered_results):
    metric_names = []
    for per_layer_results in layered_results.values():
        for metrics in per_layer_results.values():
            for metric_name in metrics.keys():
                if metric_name not in metric_names:
                    metric_names.append(metric_name)

    rows = []
    for signal_name in sorted(layered_results.keys()):
        per_layer_results = layered_results[signal_name]
        for layer in sort_layer_strategies(per_layer_results.keys()):
            row = {
                "signal": signal_name,
                "layer": layer,
                "layer_order": get_layer_order(layer),
            }
            row.update(per_layer_results[layer])
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["signal", "layer", "layer_order"] + metric_names)

    return pd.DataFrame(rows).sort_values(
        by=["signal", "layer_order", "layer"],
        kind="stable",
    )[["signal", "layer", "layer_order"] + metric_names].reset_index(drop=True)


def plot_layer_metric_trends(layer_metric_df, output_dir, metric_name="auc"):
    if layer_metric_df.empty or metric_name not in layer_metric_df.columns:
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    for signal_name, signal_df in layer_metric_df.groupby("signal", sort=True):
        signal_df = signal_df.sort_values(by=["layer_order", "layer"], kind="stable")
        plt.figure(figsize=(10, 5))
        plt.plot(signal_df["layer"], signal_df[metric_name], marker="o", linewidth=2)
        plt.title(f"{signal_name} {metric_name} by layer")
        plt.xlabel("Layer")
        plt.ylabel(metric_name.upper())
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{signal_name}_{metric_name}_by_layer.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        generated_paths.append(plot_path)

    combined_plot_path = plot_all_layer_signals(layer_metric_df, output_dir, metric_name=metric_name)
    if combined_plot_path is not None:
        generated_paths.append(combined_plot_path)

    return generated_paths


def plot_all_layer_signals(layer_metric_df, output_dir, metric_name="auc"):
    if layer_metric_df.empty or metric_name not in layer_metric_df.columns:
        return None

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 7))
    for signal_name, signal_df in layer_metric_df.groupby("signal", sort=True):
        signal_df = signal_df.sort_values(by=["layer_order", "layer"], kind="stable")
        plt.plot(signal_df["layer"], signal_df[metric_name], marker="o", linewidth=1.8, label=signal_name)

    plt.title(f"All signals {metric_name} by layer")
    plt.xlabel("Layer")
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"all_layer_signals_{metric_name}_by_layer.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return plot_path


def sanitize_feature_block(df, columns):
    if not columns:
        return df
    block = df[columns].replace([np.inf, -np.inf], np.nan)
    block = block.astype(np.float64)
    block = block.apply(
        lambda col: col.fillna(col[np.isfinite(col)].median() if np.isfinite(col).any() else 0.0)
    )
    block = block.clip(lower=-1e6, upper=1e6)
    df[columns] = block
    return df


def main():
    args = parser.parse_args()

    llava_results = load_pickle_with_numpy_compat(args.generation_file)
    image_df = pd.DataFrame().from_dict(llava_results)

    if "internal_embedding" in image_df.columns:
        image_df = image_df.rename(columns={"internal_embedding": "embedding"})
    if "embedding" not in image_df.columns:
        raise ValueError("The 'embedding' column is missing from the DataFrame.")

    image_df["norm_embedding"] = image_df["embedding"].apply(normalize_embedding)
    image_df["logdet"] = image_df["norm_embedding"].apply(
        lambda x: compute_logdet(np.matmul(x, x.T), alpha=args.jitter)
    )

    prob_values = image_df["generations_log_likelihood"].apply(get_normL1_prob)
    logdet_values = image_df["logdet"]
    prob_alpha = np.abs(logdet_values.median() / max(prob_values.median(), 1e-8))
    image_df["umpire"] = image_df.apply(lambda x: compute_probL1_logdet(x, alpha=prob_alpha), axis=1)

    base_signals = ["entropy_signal", "confidence_signal", "warning_signal", "layer_instability"]
    for signal in base_signals:
        image_df[f"uncertainty_{signal}"] = image_df.apply(lambda row, s=signal: get_row_signal(row, s), axis=1)

    image_df["uncertainty_early_stop_rate"] = image_df.apply(
        lambda row: get_row_signal(row, "early_stop_rate"), axis=1
    )
    image_df["uncertainty_generation_diversity"] = image_df.apply(
        lambda row: get_row_signal(row, "generation_diversity"), axis=1
    )
    image_df["uncertainty_online_uncertainty"] = image_df.apply(
        lambda row: get_row_signal(row, "online_uncertainty"), axis=1
    )

    all_layer_strategies = discover_layer_strategies(image_df)
    analysis_layer_strategies = select_layer_strategies_for_analysis(
        all_layer_strategies,
        scope=args.layer_analysis_scope,
    )
    layer_feature_types = [
        "mean",
        "var",
        "std",
        "max",
        "min",
        "range",
        "skew",
        "kurt",
        "norm",
        "logdet",
        "eigen_score",
        "drift",
        "delta_norm",
    ]

    print("\nExtracting layer features...")
    all_layer_feature_cols = []
    for strategy in all_layer_strategies:
        strategy_name = strategy.replace("%", "pct").replace("layer_", "")
        for feature_type in layer_feature_types:
            col_name = f"uncertainty_layer_{feature_type}_{strategy_name}"
            image_df[col_name] = image_df.apply(
                lambda x, s=strategy, t=feature_type: get_row_signal(x, f"layer_{t}_{s}"),
                axis=1,
            )
            all_layer_feature_cols.append(col_name)

    print("\n" + "=" * 60)
    print("STEP 1: Normalizing features")
    print("=" * 60)

    baseline_cols = [
        "uncertainty_entropy_signal",
        "uncertainty_confidence_signal",
        "uncertainty_warning_signal",
        "uncertainty_layer_instability",
    ]

    image_df = sanitize_feature_block(image_df, baseline_cols + all_layer_feature_cols)

    if all_layer_feature_cols:
        image_df[all_layer_feature_cols] = StandardScaler().fit_transform(image_df[all_layer_feature_cols])
    image_df[baseline_cols] = StandardScaler().fit_transform(image_df[baseline_cols])

    eval_col = "exact_match" if "exact_match" in image_df.columns else "correct"
    labels = image_df[eval_col].astype(int).values

    print("\n" + "=" * 60)
    print("STEP 2: Cross-layer feature fusion")
    print("=" * 60)

    cross_layer_features = []
    for feature_type in layer_feature_types:
        layer_cols = [
            f"uncertainty_layer_{feature_type}_{s.replace('%', 'pct').replace('layer_', '')}"
            for s in all_layer_strategies
        ]
        valid_cols = [col for col in layer_cols if col in image_df.columns]
        if valid_cols:
            col_name = f"cross_layer_{feature_type}"
            image_df[col_name] = image_df[valid_cols].mean(axis=1)
            cross_layer_features.append(col_name)

    image_df["cross_layer_energy"] = 0.5 * image_df.get("cross_layer_norm", 0.0) + 0.5 * image_df.get(
        "cross_layer_delta_norm", 0.0
    )
    image_df["cross_layer_geometry"] = 0.5 * image_df.get("cross_layer_logdet", 0.0) + 0.5 * image_df.get(
        "cross_layer_eigen_score", 0.0
    )
    for extra_col in ["cross_layer_energy", "cross_layer_geometry"]:
        if extra_col not in cross_layer_features:
            cross_layer_features.append(extra_col)

    print("\n" + "=" * 60)
    print("STEP 3: Training fusion model")
    print("=" * 60)

    final_features = baseline_cols + cross_layer_features
    X = image_df[final_features].fillna(0.0).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    clf_final = LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced")
    clf_final.fit(X_train, y_train)

    y_pred = clf_final.predict_proba(X_test)[:, 1]
    fusion_auc = roc_auc_score(y_test, y_pred)
    print(f"Fusion model AUC on held-out set: {fusion_auc:.4f}")

    image_df["final_combined_uncertainty"] = 1.0 - clf_final.predict_proba(X)[:, 1]
    image_df["unsupervised_risk"] = image_df[baseline_cols].mean(axis=1)

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating all uncertainty methods")
    print("=" * 60)

    results = {}
    baseline_methods = [
        ("umpire", image_df["umpire"].values),
        ("entropy_signal", image_df["uncertainty_entropy_signal"].values),
        ("confidence_signal", image_df["uncertainty_confidence_signal"].values),
        ("warning_signal", image_df["uncertainty_warning_signal"].values),
        ("layer_instability", image_df["uncertainty_layer_instability"].values),
        ("early_stop_rate", image_df["uncertainty_early_stop_rate"].values),
        ("generation_diversity", image_df["uncertainty_generation_diversity"].values),
        ("online_uncertainty", image_df["uncertainty_online_uncertainty"].values),
    ]

    print("\n--- Baseline Methods ---")
    for name, scores in baseline_methods:
        result = evaluate_scores_against_labels(scores, labels, eval_col)
        if result:
            results[name] = result
            print(f"{name}: AUC = {result['auc']:.4f}")

    fusion_methods = [
        ("fusion_cross_layer", image_df["final_combined_uncertainty"].values),
        ("unsupervised_risk", image_df["unsupervised_risk"].values),
    ]

    print("\n--- Fusion Methods ---")
    for name, scores in fusion_methods:
        result = evaluate_scores_against_labels(scores, labels, eval_col)
        if result:
            results[name] = result
            print(f"{name}: AUC = {result['auc']:.4f}")

    result_df = pd.DataFrame(results).T
    print(f"\n{'=' * 80}")
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{df_to_markdown_bold(result_df)}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_df.to_csv(os.path.join(args.output_dir, "uncertainty_evaluation_results.csv"))
    result_df.to_json(
        os.path.join(args.output_dir, "uncertainty_evaluation_results.json"),
        orient="index",
        indent=2,
    )

    layered_results = evaluate_layer_signals(
        image_df=image_df,
        labels=labels,
        eval_col=eval_col,
        layer_strategies=analysis_layer_strategies,
        layer_feature_types=layer_feature_types,
    )
    layer_metric_df = build_layer_metric_table(layered_results)
    layer_metric_df.to_csv(
        os.path.join(args.output_dir, "layer_signal_uncertainty_results.csv"),
        index=False,
    )
    layer_metric_df.to_json(
        os.path.join(args.output_dir, "layer_signal_uncertainty_results.json"),
        orient="records",
        indent=2,
    )
    plot_paths = plot_layer_metric_trends(
        layer_metric_df,
        args.output_dir,
        metric_name=args.layer_plot_metric,
    )
    image_df.to_pickle(os.path.join(args.output_dir, "evaluation_results_with_features.pkl"))

    print(f"\nResults saved to {args.output_dir}")
    if not layer_metric_df.empty:
        print("\nLayer signal analysis saved:")
        print(f"  Table: {os.path.join(args.output_dir, 'layer_signal_uncertainty_results.csv')}")
        print(f"  JSON:  {os.path.join(args.output_dir, 'layer_signal_uncertainty_results.json')}")
        print(f"  Plots: {len(plot_paths)} files using metric '{args.layer_plot_metric}'")

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    weights = pd.DataFrame(
        {
            "feature": final_features,
            "weight": clf_final.coef_[0],
            "abs_weight": np.abs(clf_final.coef_[0]),
        }
    ).sort_values("abs_weight", ascending=False)

    print("\nTop 10 most important features:")
    for i, (_, row) in enumerate(weights.head(10).iterrows()):
        sign = "+" if row["weight"] > 0 else "-"
        print(f"  {i + 1:2d}. {sign} {row['feature']:40} | weight: {row['weight']:.4f}")

    print("\n" + "=" * 80)
    print("FILTERING ANALYSIS")
    print("=" * 80)

    fusion_score = image_df["final_combined_uncertainty"].values
    for percentile in [90, 95, 99]:
        thr = np.percentile(fusion_score, percentile)
        mask = fusion_score < thr
        filtered_acc = np.mean(labels[mask]) if np.sum(mask) > 0 else 0.0
        print(f"  Acc @ {percentile}% filter: {filtered_acc:.4f} (retained {np.sum(mask)}/{len(labels)})")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
