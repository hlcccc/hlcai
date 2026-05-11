import pickle
import sys
sys.path.append('.')

data_path = 'output_dir/okvqa_val2014/generation_embedding/llava-1.5-13b-hf/1_0/generations_with_uncertainty.pkl'

print('Loading data...')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f'Total samples: {len(data)}')
print()

if len(data) > 0:
    sample = data[0]
    print('Sample keys:', list(sample.keys()))
    
    if 'uncertainty_info' in sample:
        ui_list = sample['uncertainty_info']
        if ui_list:
            ui = ui_list[0]
            print()
            print('Uncertainty info keys:', list(ui.keys()))
            
            if 'layer_features_by_strategy' in ui:
                print()
                print('Layer features found!')
                lfs = ui['layer_features_by_strategy']
                for strategy, features in lfs.items():
                    print(f'  {strategy}: {list(features.keys())}')
                    if features:
                        print(f'    Example values: {dict(list(features.items())[:5])}')

print('\n' + '='*80)

print('\nChecking for correct/error labels...')
correct_count = 0
error_count = 0
if len(data) > 0:
    if 'exact_match' in data[0]:
        for i, sample in enumerate(data):
            if sample['exact_match'] == 1:
                correct_count += 1
            else:
                error_count += 1
print(f'Correct samples: {correct_count}')
print(f'Error samples: {error_count}')

print('\n' + '='*80)
print('\nChecking layer feature statistics...')
if len(data) > 0 and 'uncertainty_info' in data[0]:
    sample = data[0]
    ui_list = sample['uncertainty_info']
    if ui_list:
        ui = ui_list[0]
        if 'layer_features_by_strategy' in ui:
            lfs = ui['layer_features_by_strategy']
            for strategy, features in lfs.items():
                print(f'\nStrategy: {strategy}')
                for k, v in features.items():
                    print(f'  {k}: {v}')
