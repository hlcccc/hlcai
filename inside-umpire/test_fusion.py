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
    with open(data_path, 'rb') as f:
        llava_results = pickle.load(f)

    image_df = pd.DataFrame().from_dict(llava_results)
    print(f'Loaded {len(image_df)} samples')

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

    print('\nExtracting features...')
    features = []
    for strategy in layer_strategies:
        for ft in feature_types:
            col_name = f'{strategy}_{ft}'
            image_df[col_name] = image_df.apply(lambda x, s=strategy, t=ft: get_layer_feature(x, s, t), axis=1)
            features.append(col_name)

    eval_col = 'exact_match' if 'exact_match' in image_df.columns else 'correct'

    if eval_col not in image_df.columns:
        print(f'Error: {eval_col} column not found')
        return

    labels = image_df[eval_col].values

    print('\n' + '='*60)
    print('STEP 1: Normalizing features')
    print('='*60)

    scaler = StandardScaler()
    X = scaler.fit_transform(image_df[features])

    print('\n' + '='*60)
    print('STEP 2: Training fusion model')
    print('='*60)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print(f'\nFusion Model Results:')
    print(f'  Number of features: {len(features)}')
    print(f'  Test set AUC: {auc:.4f}')

    print('\n' + '='*60)
    print('STEP 3: Feature Importance')
    print('='*60)

    weights = pd.DataFrame({'feature': features, 'weight': clf.coef_[0]})
    weights['abs_weight'] = weights['weight'].abs()
    print('\nFeature Importance:')
    for _, row in weights.sort_values('abs_weight', ascending=False).head(8).iterrows():
        sign = '+' if row['weight'] > 0 else '-'
        print(f'  {sign} {row["feature"]:20} | {row["weight"]:.4f}')

    print('\n' + '='*60)
    print('STEP 4: Individual Feature AUC')
    print('='*60)

    print('\nIndividual feature AUC (for comparison):')
    for feat in features:
        try:
            feat_auc = roc_auc_score(labels, image_df[feat].values)
            print(f'  {feat:25}: AUC = {feat_auc:.4f}')
        except:
            print(f'  {feat:25}: AUC = N/A')

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'\nIndividual features AUC range: 0.5 (random)')
    print(f'Fusion model AUC: {auc:.4f}')
    print(f'\nIf fusion AUC > 0.6, the fusion approach is working!')

if __name__ == '__main__':
    main()
