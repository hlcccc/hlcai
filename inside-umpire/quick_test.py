#!/usr/bin/env python
"""
快速测试脚本：验证层消融融合模型是否正常工作
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def main():
    data_path = 'output_dir/okvqa_val2014/generation_embedding/llava-1.5-13b-hf/1_0/generations_with_uncertainty.pkl'

    print('Loading data...')
    try:
        with open(data_path, 'rb') as f:
            llava_results = pickle.load(f)
    except FileNotFoundError:
        print(f'ERROR: Data file not found at {data_path}')
        print('Please run the generation script first:')
        print('  bash scripts/compute_umpire_with_uncertainty.sh')
        return

    image_df = pd.DataFrame().from_dict(llava_results)
    print(f'Loaded {len(image_df)} samples')

    print('\nChecking data structure...')
    if len(image_df) == 0:
        print('ERROR: No data loaded')
        return

    sample = image_df.iloc[0]
    print(f'  Columns: {list(image_df.columns)[:10]}...')

    if 'uncertainty_info' not in sample:
        print('ERROR: uncertainty_info column not found')
        return

    if not sample['uncertainty_info']:
        print('ERROR: uncertainty_info is empty')
        return

    ui = sample['uncertainty_info'][0]
    print(f'  uncertainty_info keys: {list(ui.keys())}')

    if 'layer_features_by_strategy' not in ui:
        print('ERROR: layer_features_by_strategy not found in uncertainty_info')
        return

    lfs = ui['layer_features_by_strategy']
    if not lfs:
        print('ERROR: layer_features_by_strategy is empty')
        return

    print(f'  Layer strategies: {list(lfs.keys())}')
    print(f'  Example layer features: {list(list(lfs.values())[0].keys()) if lfs else "N/A"}')

    def get_layer_feature(row, strategy, feature_name):
        if 'uncertainty_info' not in row:
            return 0.0
        for ui in row['uncertainty_info']:
            if 'layer_features_by_strategy' in ui:
                features = ui['layer_features_by_strategy'].get(strategy, {})
                return features.get(feature_name, 0.0)
        return 0.0

    layer_strategies = ['layer_6', 'layer_12', 'layer_18', 'last_layer']
    feature_types = ['logdet', 'eigen_score', 'norm', 'var']

    print('\n' + '='*60)
    print('Extracting features...')
    print('='*60)

    features = []
    for strategy in layer_strategies:
        for ft in feature_types:
            col_name = f'{strategy}_{ft}'
            image_df[col_name] = image_df.apply(lambda x, s=strategy, t=ft: get_layer_feature(x, s, t), axis=1)
            features.append(col_name)

    print(f'Extracted {len(features)} features')

    eval_col = 'exact_match' if 'exact_match' in image_df.columns else 'correct'

    if eval_col not in image_df.columns:
        print(f'ERROR: {eval_col} column not found')
        print('Available columns:', list(image_df.columns))
        return

    labels = image_df[eval_col].values
    correct_count = np.sum(labels)
    error_count = len(labels) - correct_count
    print(f'\nLabels: {correct_count} correct, {error_count} error')

    print('\n' + '='*60)
    print('STEP 1: Feature Statistics')
    print('='*60)

    for feat in features[:4]:
        vals = image_df[feat].values
        print(f'  {feat:20}: mean={vals.mean():.6f}, std={vals.std():.6f}, min={vals.min():.6f}, max={vals.max():.6f}')

    print('\n' + '='*60)
    print('STEP 2: Normalizing features')
    print('='*60)

    scaler = StandardScaler()
    X = scaler.fit_transform(image_df[features])

    print('Features normalized (Z-score)')

    print('\n' + '='*60)
    print('STEP 3: Training fusion model')
    print('='*60)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    fusion_auc = roc_auc_score(y_test, y_pred)

    print(f'\n*** Fusion Model AUC: {fusion_auc:.4f} ***')

    if fusion_auc > 0.55:
        print('✅ Fusion approach is working! (AUC > 0.55)')
    elif fusion_auc > 0.5:
        print('⚠️ Fusion shows slight improvement over random (0.5)')
    else:
        print('❌ Fusion AUC < 0.5, check data quality')

    print('\n' + '='*60)
    print('STEP 4: Individual Feature AUC')
    print('='*60)

    print('\nIndividual feature AUC (for comparison):')
    individual_aucs = []
    for feat in features:
        try:
            feat_auc = roc_auc_score(labels, image_df[feat].values)
            individual_aucs.append(feat_auc)
            print(f'  {feat:25}: AUC = {feat_auc:.4f}')
        except Exception as e:
            print(f'  {feat:25}: AUC = N/A ({e})')

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)

    avg_individual_auc = np.mean(individual_aucs) if individual_aucs else 0.5
    print(f'\nAverage individual feature AUC: {avg_individual_auc:.4f}')
    print(f'Fusion model AUC: {fusion_auc:.4f}')
    print(f'Improvement: {(fusion_auc - avg_individual_auc):.4f}')

    if fusion_auc > avg_individual_auc + 0.05:
        print('\n✅ Fusion model significantly outperforms individual features!')
    elif fusion_auc > avg_individual_auc:
        print('\n⚠️ Fusion model slightly outperforms individual features')
    else:
        print('\n❌ Fusion model does not improve over individual features')

if __name__ == '__main__':
    main()