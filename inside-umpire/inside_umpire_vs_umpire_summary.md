# inside-umpire 项目总结报告

## 1. 项目背景

`inside-umpire` 是在 `UMPIRE` 基础上扩展得到的多模态大模型不确定性量化框架。  
原始 `UMPIRE` 的核心思想是：对同一输入采样多次回答，结合回答之间的分布差异和内部 embedding 的几何量，构造训练-free 的不确定性分数。其典型实现流程是：

1. 对每个问题生成多条回答；
2. 在生成结束后保存每次回答的最后 token embedding；
3. 用多次采样回答的 embedding 几何体积和概率项计算 `umpire` 分数；
4. 在离线评估阶段计算 AUC、CECE、AURAC 等指标。

对应基线实现代码：

- 生成阶段：`pipeline/generate_and_compute_emb_hf.py`
  关键逻辑见 [pipeline/generate_and_compute_emb_hf.py](./pipeline/generate_and_compute_emb_hf.py) 第 96-167 行。
- UMPIRE 评估阶段：`pipeline/compute_umpire_and_evaluate.py`
  关键逻辑见 [pipeline/compute_umpire_and_evaluate.py](./pipeline/compute_umpire_and_evaluate.py) 第 43-103 行。

从实现上看，`UMPIRE` 的不确定性计算是 **后验式（post-hoc）** 的，即必须在回答全部生成完成后，才能利用采样回答集合与最终 embedding 计算不确定性。

---

## 2. inside-umpire 相比 UMPIRE 的核心改进

### 2.1 总体差异

| 维度 | UMPIRE | inside-umpire |
|---|---|---|
| 不确定性计算时机 | 生成结束后离线计算 | 生成过程中逐 token 在线计算 |
| 预警机制 | 无 | 有，支持 early warning / early stop |
| 隐藏状态使用方式 | 只使用最终输出 embedding | 使用中间层、多层、跨时间轨迹特征 |
| 特征设计 | 几何体积 + 序列概率 | 熵、置信度、预警信号、层轨迹统计、多层融合 |
| 层消融实验 | 不支持 | 支持具体层、比例层、池化层等多种策略 |
| 融合方式 | 单一 `umpire` 分数 | 多信号融合 + 跨层特征融合 |
| 调试与稳定性 | 无在线数值检查 | 加入 smoke test、NaN 检测、数值裁剪 |

---

## 3. 主要改进点与对应代码

### 3.1 从“生成后估计”改为“生成中在线监控”

#### UMPIRE 做法

在基线中，生成逻辑由 `model.predict_prompt_image(...)` 完成，等模型把整条回答生成完以后，再保存回答文本、log likelihood 和最终 embedding。  
代码见 [pipeline/generate_and_compute_emb_hf.py](./pipeline/generate_and_compute_emb_hf.py) 第 96-117 行。

也就是说，UMPIRE 默认只利用：

- 多条最终回答；
- 每条回答的最终 embedding；
- 每条回答的序列概率。

#### inside-umpire 做法

inside-umpire 重写了生成阶段，不再依赖“一次性生成后回看”，而是显式构造在线解码过程。  
关键代码位于 [pipeline/generate_with_uncertainty.py](./pipeline/generate_with_uncertainty.py)：

- 参数入口：第 61-78 行  
  引入了 `enable_early_warning`、`entropy_threshold`、`variance_threshold`、`warning_confidence_threshold`、`warning_drift_threshold`、`warning_risk_threshold` 等在线监控参数。
- 在线前向与采样：第 272-420 行  
  `predict_with_uncertainty()` 在 `for step_idx in range(model.max_new_tokens)` 循环中逐步执行前向传播、采样 token、更新不确定性统计量。
- 输入兼容扩展：第 148-193 行  
  为多模态模型在线解码时同步扩展 `attention_mask`、`token_type_ids`、`position_ids`，保证 LLaVA 等模型在逐 token 解码时仍然可运行。

#### 改进意义

这一变化使 inside-umpire 不再是纯后验估计，而是具备了 **推理过程监控能力**。  
在应用层面，这意味着：

- 可以在生成过程中判断模型是否逐渐进入高风险状态；
- 可以为不可靠回答提前发出预警；
- 可以进一步支持推理中断、重采样、升级模型或人工接管。

---

### 3.2 新增 Early Warning Monitor：早期预警机制

inside-umpire 的一个关键创新是把“过程不确定性”显式建模为在线预警问题。

核心代码位于 [modules/uncertainty_utils.py](./modules/uncertainty_utils.py)：

- `EarlyWarningMonitor` 类：第 38-151 行
- 动态风险更新：`update()`，第 86-137 行
- 动态 boost 机制：`_dynamic_boost()`，第 79-85 行

#### 设计思想

该模块不只看一个指标，而是综合以下过程信号：

- `entropy`：当前 token 分布熵；
- `confidence_gap`：当前 token 置信度缺口；
- `layer_spread`：不同层表示之间的分散度；
- `drift`：表示随时间的漂移幅度。

在 `update()` 中，上述信号被统一归一化后，通过加权求和得到 `risk_score`。  
然后再结合：

- 固定阈值触发；
- 历史动态异常触发；
- 连续高风险步数触发；

共同决定是否触发早停。

#### 相比 UMPIRE 的改进

UMPIRE 只在回答生成完成后给出一个“最终不确定性分数”；  
inside-umpire 则把不确定性判断前移到回答生成途中，并把它设计成一个可更新、可累积、可触发的风险监控器。

---

### 3.3 多层隐藏状态特征提取

#### UMPIRE 做法

UMPIRE 主要使用最终回答的 embedding，默认不进行系统性的层选择与层消融。

#### inside-umpire 做法

inside-umpire 引入了显式的层特征提取策略，核心代码位于 [modules/uncertainty_utils.py](./modules/uncertainty_utils.py)：

- `get_last_layer()`：第 241-245 行
- `get_mean_pooling()`：第 247-256 行
- `get_layer_by_percentage()`：第 258-264 行
- `get_eos_token_hidden_state()`：第 266-278 行
- `extract_layer_signal()`：第 281-304 行

支持的层策略包括：

- `25%`
- `50%`
- `75%`
- `last_layer`
- `eos`
- `mean_pooling`
- `layer_0, layer_1, ...`

另外，生成脚本中通过 `--eval_all_layers` 自动枚举多层策略：

- 代码见 [pipeline/generate_with_uncertainty.py](./pipeline/generate_with_uncertainty.py) 第 129-134 行。

#### 改进意义

这使得 inside-umpire 不再假设“最后一层一定最好”，而是允许系统比较：

- 哪一层最适合做不确定性估计；
- 哪类层策略最有区分能力；
- 是否需要跨层平均或跨层融合。

这也是后续层消融实验的基础。

---

### 3.4 从单点 embedding 到“层轨迹统计特征”

inside-umpire 的另一个核心改进，是不再只把单次生成末尾的 hidden state 当作一个点来处理，而是把生成过程中的层表示当作一个 **时间轨迹**。

核心代码见 [modules/uncertainty_utils.py](./modules/uncertainty_utils.py)：

- `compute_temporal_drift()`：第 335-342 行
- `compute_delta_norm()`：第 345-351 行
- `compute_layer_spread()`：第 354-365 行
- `compute_layer_statistics()`：第 368-416 行

`compute_layer_statistics()` 会对层轨迹提取以下统计特征：

- `mean`
- `var`
- `std`
- `max`
- `min`
- `range`
- `skew`
- `kurt`
- `norm`
- `logdet`
- `eigen_score`
- `drift`
- `delta_norm`

#### 相比 UMPIRE 的改进

UMPIRE 更偏向“生成结果集合的几何体积”；  
inside-umpire 进一步关注“单次生成过程中表示如何演化”，引入了轨迹稳定性与跨层结构信息。

这使 inside-umpire 能够建模：

- 中间层是否突然异常；
- 表示是否在推理过程中漂移过快；
- 不同层之间是否出现分歧。

---

### 3.5 四类在线不确定性信号

inside-umpire 当前的在线不确定性核心可以总结为四类主信号：

1. `entropy_signal`
2. `confidence_signal`
3. `warning_signal`
4. `fusion_cross_layer`

这些信号在评估代码中被显式抽取：

- `get_row_signal()`： [pipeline/evaluate_uncertainty.py](./pipeline/evaluate_uncertainty.py) 第 74-111 行
- `base_signals` 定义：第 162-164 行
- 基础方法评估：第 313-323 行
- 融合方法评估：第 332-335 行

其中：

- `entropy_signal` 对应 token 熵的时间加权平均；
- `confidence_signal` 对应 token 置信度缺口；
- `warning_signal` 对应在线预警器的风险分数；
- `fusion_cross_layer` 是跨层特征与多信号融合后的最终主分数。

#### 改进意义

这套设计不再局限于单一几何指标，而是把“不确定性”拆成：

- 预测分布不确定性；
- 过程稳定性；
- 层间结构差异；
- 综合风险评分。

因此 inside-umpire 的不确定性量化更接近一个“多视角风险模型”。

---

### 3.6 跨层融合与逻辑回归融合

在评估阶段，inside-umpire 不只比较单个层、单个特征，还增加了跨层统计和融合学习。

关键代码位于 [pipeline/evaluate_uncertainty.py](./pipeline/evaluate_uncertainty.py)：

- 自动发现层策略：第 114-125 行
- 提取所有层特征：第 193-203 行
- 特征标准化：第 205-220 行
- 跨层平均：第 229-239 行
- 几何组合特征：第 241-249 行
- 逻辑回归融合：第 251-273 行

融合步骤包括：

1. 对每种特征类型在所有层间求平均，构造 `cross_layer_*` 特征；
2. 构造 `cross_layer_energy`、`cross_layer_geometry` 等派生特征；
3. 将基础信号与跨层特征拼接；
4. 用 `LogisticRegression` 学习最终风险评分；
5. 得到 `final_combined_uncertainty`，并在结果输出中命名为 `fusion_cross_layer`。

#### 相比 UMPIRE 的改进

UMPIRE 的最终分数是固定公式计算；  
inside-umpire 则允许在保持训练-free 生成的前提下，在评估阶段学习一个更优的特征组合权重。

这也是 full tuned 实验中 `fusion_cross_layer` 超过 UMPIRE 的关键原因之一。

---

### 3.7 引入 smoke test 与数值稳定性修复

由于 inside-umpire 将不确定性前移到生成阶段，数值稳定性比 UMPIRE 更关键。  
因此项目中额外加入了在线调试与防错机制。

关键代码位于 [pipeline/generate_with_uncertainty.py](./pipeline/generate_with_uncertainty.py)：

- `--max_samples` / `--smoke_test`：第 77-78 行
- `ensure_finite_scalar()`：第 230-243 行
- `validate_uncertainty_info()`：第 246-269 行
- 在生成过程中逐步检查 NaN：第 339-360 行、第 403-410 行
- 生成结束后写 `smoke_test_summary.json`：第 571-578 行

同时，数值稳定性修复位于 [modules/uncertainty_utils.py](./modules/uncertainty_utils.py)：

- `compute_token_entropy_from_logits()`：第 212-217 行
- `compute_max_prob()`：第 220-224 行
- `compute_generation_confidence()`：第 239-245 行
- `EarlyWarningMonitor._dynamic_boost()` 中对指数输入裁剪：第 79-85 行

#### 改进意义

这部分并不是算法层面的主创新，但对于工程可用性很重要。  
它使 inside-umpire 能：

- 在长时间推理前先做小样本稳定性检查；
- 快速发现 `NaN/Inf`；
- 在早停阈值调试时快速定位错误。

相比之下，UMPIRE 原始实现更偏向“离线一次跑完”，缺少这套过程调试与在线安全防护。

---

## 4. 实验结果与效果总结

### 4.1 在当前 full tuned 实验中的表现

在 `test_output/full_gpu2_tuned_eval/uncertainty_evaluation_results.json` 中，主要结果如下：

| 方法 | AUC | AURAC |
|---|---:|---:|
| `umpire` | 0.7249 | 0.7991 |
| `entropy_signal` | 0.7205 | 0.7964 |
| `confidence_signal` | 0.7359 | 0.8046 |
| `warning_signal` | 0.7411 | 0.8109 |
| `online_uncertainty` | 0.7297 | 0.8019 |
| `fusion_cross_layer` | **0.7599** | **0.8190** |

从结果看：

1. `fusion_cross_layer` 明显超过同轮 `UMPIRE` 基线；
2. `warning_signal` 单独就具有较强的错误识别能力；
3. `confidence_signal` 与 `online_uncertainty` 稳定优于基线；
4. `layer_instability` 当前单独表现较弱，更适合作为辅助特征而不是单独主特征。

### 4.2 结论

在当前工程实现与 full tuned 参数下，inside-umpire 已经实现了：

- 从后验不确定性估计，升级为在线不确定性监控；
- 从单一几何打分，升级为多信号、多层、多时间尺度的融合估计；
- 在当前实验设置中，最终融合分数优于同轮 `UMPIRE` 基线。

---

## 5. inside-umpire 的最终改进总结

### 5.1 算法层面

inside-umpire 相比 UMPIRE 的算法改进可以归纳为：

1. 将不确定性估计前移到生成过程中；
2. 引入 Early Warning Monitor 进行动态风险判断；
3. 使用多层隐藏状态而非仅最终 embedding；
4. 使用轨迹统计特征而非单帧特征；
5. 设计多信号不确定性表示；
6. 引入跨层融合和逻辑回归融合。

### 5.2 工程层面

工程改进包括：

1. 支持多模态在线逐 token 解码；
2. 补齐 attention mask / token type / position id 的在线扩展；
3. 加入 NaN/Inf 检查与 smoke test；
4. 提供层消融实验工具与融合评估脚本。

### 5.3 项目定位

因此，inside-umpire 不再只是“UMPIRE 的复现版”，而是一个更偏向 **过程风险监控 + 多层表示分析 + 多信号融合** 的多模态不确定性量化框架。

---

## 6. 建议的最终表述

如果用于论文、答辩或项目汇报，可以将 inside-umpire 的贡献概括为：

> inside-umpire extends UMPIRE from a post-hoc response-level uncertainty estimator to an online multimodal uncertainty monitoring framework.  
> It introduces token-level early warning, multi-layer hidden-state probing, temporal representation statistics, and cross-layer fusion, thereby enabling process-aware uncertainty quantification and improving final uncertainty discrimination over the original UMPIRE baseline.

