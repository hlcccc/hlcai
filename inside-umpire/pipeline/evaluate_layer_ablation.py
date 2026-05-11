import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import sys
sys.path.append(".")

from modules.eval_utils import ROC_AUROC, compute_pearsonr, get_calibrate_ece, get_tpr_at_fpr, compute_aurac_from_image_df, df_to_markdown_bold
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--generation_file', type=str, required=True, help='Path to the generation file with uncertainty info')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')

args = parser.parse_args()

file_path = args.generation_file
if os.path.isfile(file_path):
    with open(file_path, 'rb') as r:
        llava_results = pickle.load(r)
image_df = pd.DataFrame().from_dict(llava_results)

if 'internal_embedding' in image_df.columns:
    image_df = image_df.rename(columns={'internal_embedding': 'embedding'})

print('='*100)
print('LAYER ABLATION EVALUATION')
print('='*100)

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
        elif signal_type.startswith('layer_'):
            parts = signal_type.split('_')
            if len(parts) >= 3:
                feature_name = parts[1]
                strategy = '_'.join(parts[2:])
                uncertainty_values.append(get_layer_feature_by_strategy(uncertainty_info, strategy, feature_name))

    if not uncertainty_values:
        return 0.0
    return np.mean(uncertainty_values)

# Compute baseline signals
image_df['avg_entropy'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_entropy'), axis=1)
image_df['avg_confidence'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_confidence'), axis=1)

# Define layer strategies and feature types
layer_strategies = [
    'layer_0', 'layer_3', 'layer_6', 'layer_9', 
    'layer_12', 'layer_15', 'layer_18', 'layer_21',
    'last_layer', 'mean_pooling'
]

layer_feature_types = [
    'mean', 'var', 'std', 'max', 'min', 'range', 
    'skew', 'kurt', 'norm', 'logdet', 'eigen_score'
]

# Add layer features to dataframe
print('\nAdding layer features...')
for strategy in tqdm(layer_strategies):
    strategy_name = strategy.replace('%', 'pct').replace('layer_', '')
    for feature_type in layer_feature_types:
        col_name = f'layer_{feature_type}_{strategy_name}'
        image_df[col_name] = image_df.apply(lambda x, s=strategy, t=feature_type: get_uncertainty_signal(x, f'layer_{t}_{s}'), axis=1)

# Define evaluation function
def evaluate_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

print('\n='*100)
print('EVALUATING LAYER FEATURES')
print('='*100)

eval_col = 'exact_match' if 'exact_match' in image_df.columns else 'correct'

results = []

# Evaluate baseline features
baseline_features = ['avg_entropy', 'avg_confidence']
print(f'\nBaseline features:')
for feature in baseline_features:
    if feature in image_df.columns:
        auc = evaluate_auc(image_df[eval_col], image_df[feature])
        print(f'  {feature:20}: AUC = {auc:.4f}')
        results.append({
            'feature': feature,
            'layer': 'baseline',
            'auc': auc
        })

# Evaluate all layer features
print(f'\nLayer features:')
for strategy in layer_strategies:
    strategy_name = strategy.replace('%', 'pct').replace('layer_', '')
    for feature_type in layer_feature_types:
        col_name = f'layer_{feature_type}_{strategy_name}'
        if col_name in image_df.columns:
            auc = evaluate_auc(image_df[eval_col], image_df[col_name])
            results.append({
                'feature': feature_type,
                'layer': strategy,
                'auc': auc
            })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

print('\n' + '='*100)
print('TOP PERFORMING FEATURES')
print('='*100)

# Sort by AUC
results_df_sorted = results_df.sort_values('auc', ascending=False)

print(f'\nTop 20 features:')
for i, (idx, row) in enumerate(results_df_sorted.head(20).iterrows()):
    layer_display = row['layer'] if row['layer'] != 'baseline' else 'Baseline'
    print(f'  {i+1:2d}. {layer_display:15} - {row["feature"]:12}: AUC = {row["auc"]:.4f}')

print('\n' + '='*100)
print('BEST FEATURE BY LAYER')
print('='*100)

for strategy in layer_strategies:
    layer_results = results_df[results_df['layer'] == strategy]
    if len(layer_results) > 0:
        best_result = layer_results.loc[layer_results['auc'].idxmax()]
        print(f'  {strategy:15}: Best = {best_result["feature"]:12} (AUC = {best_result["auc"]:.4f})')

print('\n' + '='*100)
print('BEST LAYER BY FEATURE TYPE')
print('='*100)

for feature_type in layer_feature_types:
    feature_results = results_df[results_df['feature'] == feature_type]
    if len(feature_results) > 0:
        best_result = feature_results.loc[feature_results['auc'].idxmax()]
        print(f'  {feature_type:12}: Best = {best_result["layer"]:15} (AUC = {best_result["auc"]:.4f})')

# Save results
os.makedirs(args.output_dir, exist_ok=True)
results_df.to_csv(os.path.join(args.output_dir, 'layer_ablation_results.csv'), index=False)
print(f'\nResults saved to {os.path.join(args.output_dir, "layer_ablation_results.csv")}')

# Create summary table for best features
print('\n' + '='*100)
print('SUMMARY: LAYER ABLATION COMPARISON')
print('='*100)

summary_data = []
for strategy in layer_strategies:
    layer_results = results_df[results_df['layer'] == strategy]
    if len(layer_results) > 0:
        best_result = layer_results.loc[layer_results['auc'].idxmax()]
        avg_auc = layer_results['auc'].mean()
        summary_data.append({
            'Layer Strategy': strategy,
            'Best Feature': best_result['feature'],
            'Best AUC': best_result['auc'],
            'Avg AUC': avg_auc
        })

summary_df = pd.DataFrame(summary_data).sort_values('Best AUC', ascending=False)

print(f'\n{"Layer Strategy":20} {"Best Feature":12} {"Best AUC":10} {"Avg AUC":10}')
print('-'*54)
for _, row in summary_df.iterrows():
    print(f'{row["Layer Strategy"]:20} {row["Best Feature"]:12} {row["Best AUC"]:10.4f} {row["Avg AUC"]:10.4f}')

summary_df.to_csv(os.path.join(args.output_dir, 'layer_ablation_summary.csv'), index=False)

print('\n' + '='*100)
print('DONE!')
print('='*100)
