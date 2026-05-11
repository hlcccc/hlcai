import os
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(".")

from modules.eval_utils import ROC_AUROC, compute_pearsonr, get_calibrate_ece, get_tpr_at_fpr, compute_aurac_from_image_df, df_to_markdown_bold
from modules.logdet_utils import normalize_embedding, get_normL1_prob, compute_logdet, compute_probL1_logdet, get_normalized_entropy, compute_eigenscore

parser = argparse.ArgumentParser()
parser.add_argument('--generation_file', type=str, required=True, help='Path to the generation file with uncertainty info')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
parser.add_argument('--jitter', type=float, default=1e-8)
parser.add_argument('--uncertainty_weight', type=float, default=0.5)

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
    return np.mean(uncertainty_values)

# === COMPUTE BASELINE FEATURES ===
image_df['norm_embedding'] = image_df['embedding'].apply(normalize_embedding)
image_df['logdet'] = image_df['norm_embedding'].apply(lambda x: compute_logdet(np.matmul(x, x.T), alpha=args.jitter))

prob_values = image_df['generations_log_likelihood'].apply(get_normL1_prob)
logdet_values = image_df['logdet']
prob_alpha = np.abs(logdet_values.median() / prob_values.median())
prob_param = prob_alpha
print("adaptive prob alpha", prob_param)

image_df['umpire'] = image_df.apply(lambda x: compute_probL1_logdet(x, alpha=prob_param), axis=1)
image_df['uncertainty_avg_entropy'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_entropy'), axis=1)
image_df['uncertainty_avg_confidence'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'avg_confidence'), axis=1)
image_df['uncertainty_early_stop_rate'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'early_stop_rate'), axis=1)
image_df['uncertainty_generation_diversity'] = image_df.apply(lambda x: get_uncertainty_signal(x, 'generation_diversity'), axis=1)

# === DEFINE LAYER STRATEGIES AND FEATURE TYPES ===
layer_strategies = [
    'layer_0', 'layer_3', 'layer_6', 'layer_9', 
    'layer_12', 'layer_15', 'layer_18', 'layer_21',
    'last_layer', 'mean_pooling'
]

layer_feature_types = [
    'mean', 'var', 'std', 'max', 'min', 'range', 
    'skew', 'kurt', 'norm', 'logdet', 'eigen_score'
]

# === EXTRACT ALL LAYER FEATURES ===
print('\nExtracting layer features...')
all_layer_feature_cols = []
for strategy in tqdm(layer_strategies):
    strategy_name = strategy.replace('%', 'pct').replace('layer_', '')
    for feature_type in layer_feature_types:
        col_name = f'uncertainty_layer_{feature_type}_{strategy_name}'
        image_df[col_name] = image_df.apply(lambda x, s=strategy, t=feature_type: get_uncertainty_signal(x, f'layer_{t}_{s}'), axis=1)
        all_layer_feature_cols.append(col_name)

# === STEP 1: FEATURE NORMALIZATION ===
print('\n' + '='*60)
print('STEP 1: Normalizing features')
print('='*60)

scaler = StandardScaler()
image_df[all_layer_feature_cols] = scaler.fit_transform(image_df[all_layer_feature_cols])

baseline_cols = ['uncertainty_avg_entropy', 'uncertainty_avg_confidence', 'uncertainty_early_stop_rate', 'uncertainty_generation_diversity']
image_df[baseline_cols] = scaler.fit_transform(image_df[baseline_cols])

# === STEP 2: PREPARE LABELS ===
eval_col = 'exact_match' if 'exact_match' in image_df.columns else 'correct'
labels = image_df[eval_col].values

# === STEP 3: CROSS-LAYER FEATURE FUSION (RECOMMENDED) ===
print('\n' + '='*60)
print('STEP 3: Cross-layer feature fusion')
print('='*60)

cross_layer_features = []
for feature_type in layer_feature_types:
    layer_cols = [f'uncertainty_layer_{feature_type}_{s.replace("%", "pct").replace("layer_", "")}' for s in layer_strategies]
    valid_cols = [col for col in layer_cols if col in image_df.columns]
    if valid_cols:
        col_name = f'cross_layer_{feature_type}'
        image_df[col_name] = image_df[valid_cols].mean(axis=1)
        cross_layer_features.append(col_name)

# === STEP 4: TRAIN FUSION MODEL ON DEV SET ===
print('\n' + '='*60)
print('STEP 4: Training fusion model on dev set')
print('='*60)

final_features = cross_layer_features + baseline_cols
X = image_df[final_features].values

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

clf_final = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
clf_final.fit(X_train, y_train)

y_pred = clf_final.predict_proba(X_test)[:, 1]
fusion_auc = roc_auc_score(y_test, y_pred)
print(f'Fusion model AUC on test set: {fusion_auc:.4f}')

image_df['final_combined_uncertainty'] = clf_final.predict_proba(X)[:, 1]

# === STEP 5: UNSUPERVISED FEATURE COMBINATION ===
print('\n' + '='*60)
print('STEP 5: Unsupervised feature combination')
print('='*60)

image_df['unsupervised_risk'] = image_df[cross_layer_features].mean(axis=1)

# === STEP 6: EVALUATE ALL METHODS ===
print('\n' + '='*60)
print('STEP 6: Evaluating all uncertainty methods')
print('='*60)

def evaluate_method(name, scores):
    valid_idx = ~np.isnan(scores)
    if np.sum(valid_idx) < 2:
        return None
    
    valid_scores = scores[valid_idx]
    valid_labels = labels[valid_idx]
    
    try:
        auc = roc_auc_score(valid_labels, valid_scores)
        cece = get_calibrate_ece(valid_scores, valid_labels, num_bins=15)
        pearsonr = np.abs(compute_pearsonr(valid_scores, valid_labels, num_bins=50)[0])
        tpr_at_01 = get_tpr_at_fpr(valid_scores, valid_labels, fpr=0.1)
        tpr_at_001 = get_tpr_at_fpr(valid_scores, valid_labels, fpr=0.01)
        aurac = compute_aurac_from_image_df(pd.DataFrame({'score': valid_scores, eval_col: valid_labels}), 'score', eval_col)
        
        return {
            'auc': auc,
            'cece': cece,
            'pearsonr': pearsonr,
            'tpr_at_0.1_fpr': tpr_at_01,
            'tpr_at_0.01_fpr': tpr_at_001,
            'aurac': aurac
        }
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return None

# === EVALUATE KEY METHODS ===
results = {}

# Baseline methods
baseline_methods = [
    ('umpire', image_df['umpire'].values),
    ('entropy', image_df['uncertainty_avg_entropy'].values),
    ('confidence', image_df['uncertainty_avg_confidence'].values),
    ('early_stop_rate', image_df['uncertainty_early_stop_rate'].values),
    ('generation_diversity', image_df['uncertainty_generation_diversity'].values),
]

print('\n--- Baseline Methods ---')
for name, scores in baseline_methods:
    result = evaluate_method(name, scores)
    if result:
        results[name] = result
        print(f'{name}: AUC = {result["auc"]:.4f}')

# Fusion methods
fusion_methods = [
    ('fusion_cross_layer', image_df['final_combined_uncertainty'].values),
    ('unsupervised_risk', image_df['unsupervised_risk'].values),
]

print('\n--- Fusion Methods ---')
for name, scores in fusion_methods:
    result = evaluate_method(name, scores)
    if result:
        results[name] = result
        print(f'{name}: AUC = {result["auc"]:.4f}')

# === PRINT RESULTS TABLE ===
result_df = pd.DataFrame(results).T
print(f"\n{'='*80}")
print("FINAL EVALUATION RESULTS")
print('='*80)
print(f"\n{df_to_markdown_bold(result_df)}")

# === SAVE RESULTS ===
os.makedirs(args.output_dir, exist_ok=True)
result_df.to_csv(os.path.join(args.output_dir, 'uncertainty_evaluation_results.csv'))
image_df.to_pickle(os.path.join(args.output_dir, 'evaluation_results_with_features.pkl'))

print(f"\nResults saved to {args.output_dir}")

# === FEATURE IMPORTANCE ANALYSIS ===
print('\n' + '='*80)
print('FEATURE IMPORTANCE ANALYSIS')
print('='*80)

weights = pd.DataFrame({
    'feature': final_features,
    'weight': clf_final.coef_[0],
    'abs_weight': np.abs(clf_final.coef_[0])
}).sort_values('abs_weight', ascending=False)

print('\nTop 10 most important features:')
for i, (_, row) in enumerate(weights.head(10).iterrows()):
    sign = '+' if row['weight'] > 0 else '-'
    print(f'  {i+1:2d}. {sign} {row["feature"]:40} | weight: {row["weight"]:.4f}')

# === FILTERING ANALYSIS ===
print('\n' + '='*80)
print('FILTERING ANALYSIS')
print('='*80)

fusion_score = image_df['final_combined_uncertainty'].values

for percentile in [90, 95, 99]:
    thr = np.percentile(fusion_score, percentile)
    mask = fusion_score < thr
    filtered_acc = np.mean(labels[mask]) if np.sum(mask) > 0 else 0.0
    print(f'  Acc @ {percentile}% filter: {filtered_acc:.4f} (retained {np.sum(mask)}/{len(labels)})')

print('\n' + '='*80)
print('DONE!')
print('='*80)