#!/usr/bin/env python
"""小规模测试层消融实验代码"""
import sys
sys.path.append(".")

import torch
import numpy as np

# 测试新的特征提取函数
from modules.layer_ablation_utils import (
    compute_logdet,
    compute_eigen_score,
    compute_incoherence_score,
    compute_layer_ablation_features
)

from modules.uncertainty_utils import (
    extract_layer_signal,
    compute_layer_statistics
)

print("=== 测试层消融特征提取 ===")

# 创建模拟数据
np.random.seed(42)
hidden_state = torch.tensor(np.random.randn(10, 768).astype(np.float32))

print("\n1. 测试 compute_layer_ablation_features:")
features = compute_layer_ablation_features(hidden_state)
print(f"   提取的特征数: {len(features)}")
print(f"   特征列表: {list(features.keys())}")
print(f"   特征值示例:")
for k, v in features.items():
    print(f"     {k}: {v:.6f}")

print("\n2. 测试 compute_logdet:")
logdet_val = compute_logdet(hidden_state)
print(f"   logdet: {logdet_val:.6f}")

print("\n3. 测试 compute_eigen_score:")
eigen_val = compute_eigen_score(hidden_state)
print(f"   eigen_score: {eigen_val:.6f}")

print("\n4. 测试 compute_incoherence_score:")
incoherence_val = compute_incoherence_score(hidden_state)
print(f"   incoherence: {incoherence_val:.6f}")

print("\n5. 测试 extract_layer_signal 支持具体层索引:")
hidden_tuple = (
    torch.randn(2, 768),  # layer 0
    torch.randn(2, 768),  # layer 1
    torch.randn(2, 768),  # layer 2
    torch.randn(2, 768),  # layer 3
)
layer_0 = extract_layer_signal(hidden_tuple, 'layer_0')
layer_3 = extract_layer_signal(hidden_tuple, 'layer_3')
last_layer = extract_layer_signal(hidden_tuple, 'last_layer')
print(f"   layer_0 shape: {layer_0.shape if layer_0 is not None else 'None'}")
print(f"   layer_3 shape: {layer_3.shape if layer_3 is not None else 'None'}")
print(f"   last_layer shape: {last_layer.shape if last_layer is not None else 'None'}")

print("\n✅ 所有测试通过！")
