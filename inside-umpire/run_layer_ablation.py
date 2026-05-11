import pickle
import numpy as np
import sys
import os
sys.path.append('.')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_data(data_path):
    """Load generation data"""
    print(f'Loading data from {data_path}...')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded {len(data)} samples')
    return data

def extract_features(data, strategy='layer_12'):
    """Extract features and labels"""
    print(f'\nExtracting features for strategy: {strategy}')
    
    features_list = []
    labels = []
    
    for sample in data:
        if 'exact_match' not in sample:
            continue
            
        label = sample['exact_match']
        
        if 'uncertainty_info' not in sample or not sample['uncertainty_info']:
            continue
            
        ui = sample['uncertainty_info'][0]
        
        if 'layer_features_by_strategy' not in ui:
            continue
            
        lfs = ui['layer_features_by_strategy']
        
        if strategy not in lfs:
            continue
            
        features = lfs[strategy]
        
        feature_vector = [
            features.get('mean', 0.0),
            features.get('var', 0.0),
            features.get('std', 0.0),
            features.get('max', 0.0),
            features.get('min', 0.0),
            features.get('range', 0.0),
            features.get('skew', 0.0),
            features.get('kurt', 0.0),
            features.get('norm', 0.0),
            features.get('logdet', 0.0),
            features.get('eigen_score', 0.0),
        ]
        
        features_list.append(feature_vector)
        labels.append(label)
    
    features_array = np.array(features_list)
    labels_array = np.array(labels)
    
    print(f'Extracted {len(features_list)} samples')
    print(f'  Positive (correct): {np.sum(labels_array)}')
    print(f'  Negative (error): {len(labels_array) - np.sum(labels_array)}')
    
    return features_array, labels_array

def evaluate_single_feature(features, labels, feature_idx, feature_name):
    """Evaluate a single feature"""
    try:
        feature_values = features[:, feature_idx]
        
        # Check if all values are the same
        if np.all(feature_values == feature_values[0]):
            return {
                'name': feature_name,
                'auc': 0.5,
                'mean_correct': 0.0,
                'mean_error': 0.0,
                'std_correct': 0.0,
                'std_error': 0.0
            }
        
        auc = roc_auc_score(labels, feature_values)
        
        correct_idx = labels == 1
        error_idx = labels == 0
        
        mean_correct = np.mean(feature_values[correct_idx]) if np.any(correct_idx) else 0.0
        mean_error = np.mean(feature_values[error_idx]) if np.any(error_idx) else 0.0
        std_correct = np.std(feature_values[correct_idx]) if np.any(correct_idx) else 0.0
        std_error = np.std(feature_values[error_idx]) if np.any(error_idx) else 0.0
        
        return {
            'name': feature_name,
            'auc': auc,
            'mean_correct': mean_correct,
            'mean_error': mean_error,
            'std_correct': std_correct,
            'std_error': std_error
        }
    except Exception as e:
        return {
            'name': feature_name,
            'auc': 0.5,
            'mean_correct': 0.0,
            'mean_error': 0.0,
            'std_correct': 0.0,
            'std_error': 0.0
        }

def evaluate_all_features(features, labels):
    """Evaluate all features"""
    feature_names = [
        'mean', 'var', 'std', 'max', 'min', 
        'range', 'skew', 'kurt', 'norm', 
        'logdet', 'eigen_score'
    ]
    
    results = []
    for i, name in enumerate(feature_names):
        result = evaluate_single_feature(features, labels, i, name)
        results.append(result)
    
    return results

def evaluate_combined_model(features, labels):
    """Evaluate combined model using logistic regression"""
    try:
        if len(np.unique(labels)) < 2:
            return 0.5
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(features_scaled, labels)
        
        predictions = clf.predict_proba(features_scaled)[:, 1]
        auc = roc_auc_score(labels, predictions)
        
        return auc
    except Exception as e:
        return 0.5

def main():
    data_path = 'output_dir/okvqa_val2014/generation_embedding/llava-1.5-13b-hf/1_0/generations_with_uncertainty.pkl'
    
    if not os.path.exists(data_path):
        print(f'Data path not found: {data_path}')
        return
    
    data = load_data(data_path)
    
    strategies = [
        'layer_0', 'layer_3', 'layer_6', 'layer_9', 
        'layer_12', 'layer_15', 'layer_18', 'layer_21',
        'last_layer', 'mean_pooling'
    ]
    
    print('='*100)
    print('LAYER ABLATION EXPERIMENT RESULTS')
    print('='*100)
    
    all_results = {}
    
    for strategy in strategies:
        try:
            features, labels = extract_features(data, strategy)
            
            if len(features) < 10:
                print(f'Skipping {strategy}: not enough data')
                continue
            
            # Evaluate individual features
            feature_results = evaluate_all_features(features, labels)
            
            # Evaluate combined model
            combined_auc = evaluate_combined_model(features, labels)
            
            # Sort features by AUC
            feature_results.sort(key=lambda x: x['auc'], reverse=True)
            
            all_results[strategy] = {
                'features': feature_results,
                'combined_auc': combined_auc
            }
            
            # Print results
            print(f'\n{"="*60}')
            print(f'STRATEGY: {strategy}')
            print(f'{"="*60}')
            
            print(f'\nCombined model AUC: {combined_auc:.4f}')
            
            print(f'\nTop 5 features:')
            for i, result in enumerate(feature_results[:5]):
                print(f'  {i+1}. {result["name"]:12} AUC: {result["auc"]:.4f}')
                print(f'      Correct: {result["mean_correct"]:.6f} ± {result["std_correct"]:.6f}')
                print(f'      Error:   {result["mean_error"]:.6f} ± {result["std_error"]:.6f}')
            
        except Exception as e:
            print(f'Error processing {strategy}: {e}')
            continue
    
    # Print summary
    print('\n' + '='*100)
    print('SUMMARY: BEST STRATEGIES')
    print('='*100)
    
    summary_data = []
    for strategy, result in all_results.items():
        best_feature_auc = max([f['auc'] for f in result['features']]) if result['features'] else 0.5
        summary_data.append({
            'strategy': strategy,
            'best_feature_auc': best_feature_auc,
            'combined_auc': result['combined_auc']
        })
    
    # Sort by best feature AUC
    summary_data.sort(key=lambda x: x['best_feature_auc'], reverse=True)
    
    print(f'\n{"Strategy":20} {"Best Feature AUC":20} {"Combined AUC":15}')
    print('-'*55)
    for item in summary_data:
        print(f'{item["strategy"]:20} {item["best_feature_auc"]:20.4f} {item["combined_auc"]:15.4f}')
    
    print('\nDone!')

if __name__ == '__main__':
    main()
