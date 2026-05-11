# 层消融实验使用指南

## 概述

本项目实现了完整的层消融实验功能，用于评估 LLaVA 模型不同层的隐藏状态对于不确定性量化的有效性。

## 修改总结

### 1. 新增/修改的文件

- **`modules/uncertainty_utils.py`**: 新增丰富的特征提取函数
  - `compute_logdet()`: 计算协方差矩阵的 log-determinant (INSIDE 方法)
  - `compute_eigen_score()`: 计算特征值分数
  - `compute_layer_statistics()`: 提取 11 种统计特征
  
- **`pipeline/generate_with_uncertainty.py`**: 修改以支持新特征
  - 使用具体层索引 (layer_0, layer_3, ..., layer_21)
  - 保存所有层的统计特征
  
- **`pipeline/evaluate_layer_ablation.py`**: 新增专业评估脚本
  - 评估所有层特征的 AUC
  - 生成详细的分析报告
  
- **`run_layer_ablation.py`**: 新增独立实验工具
  - 可直接运行的层消融实验
  - 包含组合模型评估
  
- **`pipeline/evaluate_uncertainty.py`**: 修改以支持新特征
  - 更新特征提取逻辑

## 使用方法

### 方法一: 运行完整流程

```bash
cd /ai/teacher/hlc/inside-umpire

# 1. 首先确保数据存在（或者重新生成）
python pipeline/generate_with_uncertainty.py \
    --model_path llava-hf/llava-1.5-13b-hf \
    --question_file data/okvqa/okvqa_processed.jsonl \
    --image_folder /ai/teacher/ssz/all_data/mqa/OKVQA/val2014 \
    --outdir output_dir/okvqa_val2014/generation_embedding/llava-1.5-13b-hf/1_0 \
    --num_generations_per_prompt 3 \
    --jitter 1e-8 \
    --enable_early_warning False \
    --layer_strategy last_layer \
    --eval_all_layers True

# 2. 运行专门的层消融评估
python pipeline/evaluate_layer_ablation.py \
    --generation_file output_dir/okvqa_val2014/generation_embedding/llava-1.5-13b-hf/1_0/generations_with_uncertainty.pkl \
    --output_dir output_dir/okvqa_val2014/layer_ablation_results
```

### 方法二: 使用独立实验工具

```bash
cd /ai/teacher/hlc/inside-umpire
python run_layer_ablation.py
```

这会直接加载已生成的数据并运行完整的层消融实验。

## 特征说明

### 提取的统计特征

每个层策略会提取以下 11 种特征：

| 特征名称 | 说明 |
|---------|------|
| `mean` | 隐藏状态的均值 |
| `var` | 隐藏状态的方差 |
| `std` | 隐藏状态的标准差 |
| `max` | 隐藏状态的最大值 |
| `min` | 隐藏状态的最小值 |
| `range` | 隐藏状态的范围 (max-min) |
| `skew` | 偏度 |
| `kurt` | 峰度 |
| `norm` | L2 范数 |
| `logdet` | 协方差矩阵的 log-determinant (INSIDE) |
| `eigen_score` | 特征值分数 |

### 评估的层策略

- `layer_0`: 第 0 层
- `layer_3`: 第 3 层
- `layer_6`: 第 6 层
- `layer_9`: 第 9 层
- `layer_12`: 第 12 层
- `layer_15`: 第 15 层
- `layer_18`: 第 18 层
- `layer_21`: 第 21 层
- `last_layer`: 最后一层
- `mean_pooling`: 所有层均值池化

## 输出说明

### 层消融评估输出

运行 `evaluate_layer_ablation.py` 会输出：

1. **Top 20 features**: 表现最好的 20 个特征
2. **Best feature by layer**: 每层的最佳特征
3. **Best layer by feature type**: 每种特征类型的最佳层
4. **Layer ablation comparison**: 所有层的综合对比

### 保存的文件

- `layer_ablation_results.csv`: 所有特征的 AUC 结果
- `layer_ablation_summary.csv`: 每层的总结信息

## 预期结果

### 理想情况下应该看到

1. **不同层的 AUC 有差异** - 某些层应该比其他层表现更好
2. **logdet/eigen_score 有较好表现** - 这些是理论上更有效的指标
3. **中层可能表现最好** - 中层通常包含更好的语义信息

### 如果 AUC 仍然 ~0.5

如果所有特征 AUC 仍然接近 0.5，可能的原因：

1. **数据质量问题** - 正确/错误样本的特征分布没有差异
2. **特征提取方式问题** - 当前只是取平均，可以尝试更复杂的方法
3. **需要更复杂的模型** - 单特征可能不够，需要多特征组合

## 下一步建议

### 如果某些特征有 AUC > 0.6

1. 选择表现最好的层和特征类型
2. 将其集成到最终的 `combined_uncertainty` 计算中
3. 验证性能提升

### 如果需要更好的特征

可以尝试：

1. **注意力特征** - 利用注意力权重的统计信息
2. **序列级别特征** - 不仅看最终状态，还要看生成过程中的变化
3. **多特征组合** - 使用逻辑回归等模型组合多个特征

## 技术细节

### 特征提取过程

```python
# 在生成过程中
for each generation:
    hidden_states = model.generate(..., output_hidden_states=True)
    
    for strategy in layer_strategies:
        extracted = extract_layer_signal(hidden_states, strategy)
        features = compute_layer_statistics(extracted)
        save(features)
```

### AUC 计算

```python
# 对于每个特征
auc = roc_auc_score(labels, feature_values)
```

## 故障排除

### 找不到数据文件

确保 `output_dir` 路径正确，数据已经生成。

### 特征全为 0

检查生成过程中 `eval_all_layers` 是否设为 `True`。

### sklearn 导入错误

```bash
pip install scikit-learn
```

## 联系方式

如有问题，查看代码注释或运行日志信息。
