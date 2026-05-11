import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import sys
sys.path.append(".")

from modules.eval_utils import ROC_AUROC, compute_pearsonr, get_calibrate_ece, get_tpr_at_fpr, compute_aurac_from_image_df, \
    df_to_markdown_bold
from modules.logdet_utils import normalize_embedding, get_normL1_prob, compute_logdet, \
    compute_probL1_logdet, get_normalized_entropy, compute_eigenscore
from modules.uncertainty_utils import compute_uncertainty_score

parser = argparse.ArgumentParser()
parser.add_argument('--generation_file', type=str, required=True,
                   help='Path to the generation file with uncertainty info')
parser.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save the output files')
parser.add_argument('--jitter', type=float, default=1e-8,
                   help='Jitter value for numerical stability in logdet computation')
parser.add_argument('--uncertainty_weight', type=float, default=0.5,
                   help='Weight for combining entropy and variance in uncertainty score')

args = parser.parse_args()

file_path = args.generation_file
if os.path.isfile(file_path):
    with open(file_path, 'rb') as r:
        llava_results = pickle.load(r)
image_df = pd.DataFrame().from_dict(llava_results)

if 'internal_embedding' in image_df.columns:
    image_df = image_df.rename(columns={'internal_embedding': 'embedding'})
if 'embedding' not in image_df.columns:
    raise ValueError("The 'embedding' column is missing from the DataFrame.")

def compute_avg_token_entropy(uncertainty_info):
    if not uncertainty_info or 'token_entropies' not in uncertainty_info:
        return 0.0
    entropies = uncertainty_info['token_entropies']
    if not entropies:
        return 0.0
    return np.mean(entropies)

def compute_avg_confidence(uncertainty_info):
    if not uncertainty_info or 'confidence_scores' not in uncertainty_info:
        return 1.0
    confidences = uncertainty_info['confidence_scores']
    if not confidences:
        return 1.0
    return np.mean(confidences)

def compute_early_stop_indicator(uncertainty_info):
    if not uncertainty_info:
        return 0.0
    return 1.0 if uncertainty_info.get('early_stop_triggered', False) else 0.0

def compute_generation_diversity(generations_text):
    if not generations_text or len(generations_text) < 2:
        return 0.0
    unique_generations = len(set(generations_text))
    return unique_generations / len(generations_text)

image_df['norm_embedding'] = image_df['embedding'].apply(normalize_embedding)

image_df['logdet'] = image_df['norm_embedding'].apply(lambda x: compute_logdet(np.matmul(x, x.T), alpha=args.jitter))

prob_values = image_df['generations_log_likelihood'].apply(get_normL1_prob)
logdet_values = image_df['logdet']
prob_alpha = np.abs(logdet_values.median() / prob_values.median())
prob_param = prob_alpha
print("adaptive prob alpha", prob_param)

image_df['umpire'] = image_df.apply(lambda x: compute_probL1_logdet(x, alpha=prob_param), axis=1)

def get_layer_feature_by_strategy(uncertainty_info, strategy, feature_name):
    if not uncertainty_info or 'layer_features_by_strategy' not in uncertainty_info:
        return 0.0
    features = uncertainty_info['layer_features_by_strategy'].get(strategy, {})
    return features.get(feature_name, 0.0)

def get_uncertainty_signal(row, signal_type='avg_entropy'):
    if 'uncertainty_info' not in row:
        return 0.0

    uncertainty_values = []
    for uncertainty_info in row['uncertainty_info']:
        if signal_type == 'avg_entropy':
            uncertainty_values.append(compute_avg_token_entropy(uncertainty_info))
        elif signal_type == 'avg_confidence':
            uncertainty_values.append(compute_avg_confidence(uncertainty_info))
        elif signal_type == 'early_stop_rate':
            uncertainty_values.append(compute_early_stop_indicator(uncertainty_info))
        elif signal_type == 'generation_diversity':
            uncertainty_values.append(compute_generation_diversity(row.get('generations_text', [])))
        elif signal_type.startswith('layer_'):
            parts = signal_type.split('_')
            if len(parts) >= 3:
                feature_name = parts[1]
                strategy = '_'.join(parts[2:])
                uncertainty_values.append(get_layer_feature_by_strategy(uncertainty_info, strategy, feature_name))

    if not uncertainty_values:
        return 0.0

    if signal_type in ['avg_entropy', 'early_stop_rate', 'generation_diversity'] or signal_type.startswith('layer_'):
        return np.mean(uncertainty_values)
    elif signal_type == 'avg_confidence':
        return np.mean(uncertainty_values)

def get_per_sample_uncertainty(row, alpha=0.5):
    avg_entropy = get_uncertainty_signal(row, 'avg_entropy')
    early_stop_rate = get_uncertainty_signal(row, 'early_stop_rate')

    avg_entropy_normalized = avg_entropy / (np.log(50000) + 1e-8)
    combined_uncertainty = alpha * avg_entropy_normalized + (1 - alpha) * early_stop_rate

    return combined_uncertainty

image_df['uncertainty_avg_entropy'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_entropy'), axis=1)
image_df['uncertainty_avg_confidence'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_confidence'), axis=1)
image_df['uncertainty_early_stop_rate'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'early_stop_rate'), axis=1)
image_df['uncertainty_generation_diversity'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'generation_diversity'), axis=1)
image_df['combined_uncertainty'] = image_df.apply(lambda x: get_per_sample_uncertainty(x, alpha=args.uncertainty_weight), axis=1)

layer_strategies = [
    'layer_0', 'layer_3', 'layer_6', 'layer_9', 
    'layer_12', 'layer_15', 'layer_18', 'layer_21',
    'last_layer', 'mean_pooling'
]
layer_feature_types = [
    'var', 'std', 'norm', 'eigen_score', 'logdet', 'incoherence',
    'mean', 'max', 'min', 'range', 'skew', 'kurt', 'spectral_norm'
]

for strategy in layer_strategies:
    strategy_name = strategy.replace('%', 'pct').replace('layer_', '')
    for feature_type in layer_feature_types:
        col_name = f'uncertainty_layer_{feature_type}_{strategy_name}'
        image_df[col_name] = image_df.apply(lambda x, s=strategy, t=feature_type: get_uncertainty_signal(x, f'layer_{t}_{s}'), axis=1)

layer_columns = []
for feature_type in layer_feature_types:
    for strategy in layer_strategies:
        strategy_name = strategy.replace('%', 'pct').replace('layer_', '')
        layer_columns.append(f'uncertainty_layer_{feature_type}_{strategy_name}')

unc_col_to_eval_list = ['umpire', 'uncertainty_avg_entropy', 'uncertainty_avg_confidence',
                        'uncertainty_early_stop_rate', 'uncertainty_generation_diversity',
                        'combined_uncertainty'] + layer_columns
conf_col_to_eval_list = []

def update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15, eval_col='exact_match'):
    image_correct_df = image_df.loc[image_df[eval_col] == 1]
    image_wrong_df = image_df.loc[image_df[eval_col] == 0]

    result_dict = {}
    for col in conf_col_to_eval_list + unc_col_to_eval_list:
        valid_mask = image_df[col].notna() & image_df[eval_col].notna()
        valid_data_col = image_df.loc[valid_mask, col]
        valid_data_eval = image_df.loc[valid_mask, eval_col]

        if len(valid_data_col) < 2:
            print(f"Skipping {col}: only {len(valid_data_col)} valid data points")
            continue

        if col in conf_col_to_eval_list:
            auc = ROC_AUROC(image_wrong_df[col], image_correct_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece', is_uncertainty=False)
            tpr_at_10_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=False, eval_col=eval_col)
            is_uncertainty = False
        else:
            auc = ROC_AUROC(image_correct_df[col], image_wrong_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece')
            tpr_at_10_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=True, eval_col=eval_col)
            is_uncertainty = True

        pearsonr = np.abs(compute_pearsonr(valid_data_col, valid_data_eval, num_bins=cpc_num_bins)[0])
        result_dict[col] = {
            'auc': auc,
            'cece': cece,
            'pearsonr': pearsonr,
            'tpr_at_0.1_fpr': tpr_at_10_fpr,
            'tpr_at_0.01_fpr': tpr_at_1_fpr,
            'aurac': aurac
        }
    return result_dict

result_dict = update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15)
result_df = pd.DataFrame().from_dict(result_dict, orient='index')
result_df = result_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)
print(df_to_markdown_bold(result_df))

early_stop_summary = {
    'total_samples': int(len(image_df)),
    'samples_with_early_stop': int((image_df['uncertainty_early_stop_rate'] > 0).sum()),
    'avg_early_stop_rate': float(image_df['uncertainty_early_stop_rate'].mean()),
    'avg_token_entropy': float(image_df['uncertainty_avg_entropy'].mean()),
    'avg_confidence': float(image_df['uncertainty_avg_confidence'].mean())
}
print("\n=== Early Warning Summary ===")
for key, value in early_stop_summary.items():
    print(f"{key}: {value}")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

result_json_file = os.path.join(args.output_dir, 'uncertainty_evaluation_results.json')
result_df.to_json(result_json_file, orient='index', indent=4)

early_stop_json_file = os.path.join(args.output_dir, 'early_warning_summary.json')
import json
with open(early_stop_json_file, 'w') as f:
    json.dump(early_stop_summary, f, indent=4)

print(f"\nResults saved to {args.output_dir}")
