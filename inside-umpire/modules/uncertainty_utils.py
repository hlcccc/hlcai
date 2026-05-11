import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy

def compute_token_entropy(logits: torch.Tensor) -> float:
    entropy = compute_entropy(logits)
    return entropy.mean().item()

def compute_sequence_entropy(scores_tuple: Tuple[torch.Tensor]) -> List[float]:
    entropies = []
    for logits in scores_tuple:
        entropy = compute_token_entropy(logits)
        entropies.append(entropy)
    return entropies

def compute_hidden_states_variance(hidden_states: torch.Tensor) -> torch.Tensor:
    variance = torch.var(hidden_states, dim=0)
    return variance.mean()

def compute_layer_wise_variance(hidden_states_tuple: tuple) -> List[float]:
    variances = []
    for hidden_state in hidden_states_tuple:
        if isinstance(hidden_state, torch.Tensor):
            var = hidden_state.var(dim=0).mean().item()
            variances.append(var)
    return variances

def compute_perplexity(logits: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.cross_entropy(logits, logits))

def compute_token_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return probs

def compute_max_prob(logits: torch.Tensor) -> float:
    probs = compute_token_probabilities(logits)
    max_prob = probs.max(dim=-1)[0].mean().item()
    return max_prob

def compute_uncertainty_score(entropy: float, variance: float, alpha: float = 0.5) -> float:
    normalized_entropy = entropy / (np.log(50000) + 1e-8)
    normalized_variance = variance / (variance + 1e-8)
    uncertainty_score = alpha * normalized_entropy + (1 - alpha) * normalized_variance
    return uncertainty_score

class EarlyWarningMonitor:
    def __init__(
        self,
        entropy_threshold: float = 0.7,
        variance_threshold: float = 0.5,
        window_size: int = 3,
        consecutive_threshold: int = 2
    ):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self.consecutive_threshold = consecutive_threshold
        self.entropy_history = []
        self.variance_history = []
        self.early_stop_triggered = False
        self.stop_reason = None

    def update(self, entropy: float, variance: Optional[float] = None):
        self.entropy_history.append(entropy)
        if variance is not None:
            self.variance_history.append(variance)

        if len(self.entropy_history) > self.window_size:
            self.entropy_history.pop(0)
        if len(self.variance_history) > self.window_size:
            self.variance_history.pop(0)

    def check_early_stop(self) -> Tuple[bool, Optional[str]]:
        if len(self.entropy_history) < self.consecutive_threshold:
            return False, None

        recent_entropies = self.entropy_history[-self.consecutive_threshold:]
        if all(e > self.entropy_threshold for e in recent_entropies):
            self.early_stop_triggered = True
            self.stop_reason = f"High entropy: {recent_entropies}"
            return True, self.stop_reason

        if self.variance_history:
            recent_variances = self.variance_history[-self.consecutive_threshold:]
            if all(v > self.variance_threshold for v in recent_variances):
                self.early_stop_triggered = True
                self.stop_reason = f"High variance: {recent_variances}"
                return True, self.stop_reason

        return False, None

    def get_current_uncertainty(self) -> dict:
        current_entropy = self.entropy_history[-1] if self.entropy_history else 0.0
        current_variance = self.variance_history[-1] if self.variance_history else 0.0
        avg_entropy = np.mean(self.entropy_history) if self.entropy_history else 0.0
        avg_variance = np.mean(self.variance_history) if self.variance_history else 0.0

        return {
            'current_entropy': current_entropy,
            'current_variance': current_variance,
            'avg_entropy': avg_entropy,
            'avg_variance': avg_variance,
            'early_stop_triggered': self.early_stop_triggered,
            'stop_reason': self.stop_reason
        }

    def reset(self):
        self.entropy_history = []
        self.variance_history = []
        self.early_stop_triggered = False
        self.stop_reason = None

class AdaptiveThreshold:
    def __init__(self, initial_threshold: float = 0.7, adjustment_factor: float = 1.1):
        self.initial_threshold = initial_threshold
        self.adjustment_factor = adjustment_factor
        self.current_threshold = initial_threshold
        self.history = []

    def update(self, uncertainty_value: float):
        self.history.append(uncertainty_value)
        if len(self.history) >= 10:
            recent_avg = np.mean(self.history[-10:])
            if recent_avg < self.current_threshold * 0.8:
                self.current_threshold *= self.adjustment_factor
            elif recent_avg > self.current_threshold * 1.2:
                self.current_threshold /= self.adjustment_factor

    def get_threshold(self) -> float:
        return self.current_threshold

def compute_semantic_drift(hidden_states: torch.Tensor) -> float:
    if len(hidden_states) < 2:
        return 0.0

    drifts = []
    for i in range(1, len(hidden_states)):
        if isinstance(hidden_states[i], torch.Tensor) and isinstance(hidden_states[i-1], torch.Tensor):
            drift = torch.norm(hidden_states[i] - hidden_states[i-1]).item()
            drifts.append(drift)

    return np.mean(drifts) if drifts else 0.0

def compute_generation_confidence(entropy: float, max_prob: float) -> float:
    confidence = 1.0 - (entropy / (np.log(50000) + 1e-8))
    confidence = confidence * 0.5 + max_prob * 0.5
    return confidence

def extract_intermediate_states(outputs, model_type='llava') -> dict:
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states
    elif hasattr(outputs, 'decoder_hidden_states'):
        hidden_states = outputs.decoder_hidden_states
    else:
        hidden_states = None

    if hasattr(outputs, 'scores'):
        logits = outputs.scores
    elif hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = None

    sequences = outputs.sequences if hasattr(outputs, 'sequences') else None

    return {
        'hidden_states': hidden_states,
        'logits': logits,
        'sequences': sequences
    }

def get_layer_by_percentage(hidden_states_tuple: tuple, percentage: float) -> torch.Tensor:
    """
    根据百分比获取特定层的隐藏状态
    percentage: 0.0-1.0，例如0.25表示前25%层，0.5表示中间层，0.75表示后25%层
    """
    if not hidden_states_tuple:
        return None
    
    num_layers = len(hidden_states_tuple)
    layer_index = int(num_layers * percentage)
    
    # 确保索引有效
    layer_index = max(0, min(layer_index, num_layers - 1))
    
    return hidden_states_tuple[layer_index]

def get_last_layer(hidden_states_tuple: tuple) -> torch.Tensor:
    """获取最后一层的隐藏状态"""
    if not hidden_states_tuple:
        return None
    return hidden_states_tuple[-1]

def get_mean_pooling(hidden_states_tuple: tuple) -> torch.Tensor:
    """对所有层进行mean pooling"""
    if not hidden_states_tuple:
        return None
    
    valid_layers = [h for h in hidden_states_tuple if isinstance(h, torch.Tensor)]
    if not valid_layers:
        return None
    
    stacked = torch.stack(valid_layers)
    return stacked.mean(dim=0)

def get_eos_token_hidden_state(hidden_states_tuple: tuple, sequence_length: int) -> torch.Tensor:
    """
    获取最后一个token（EOS）的隐藏状态
    从最后一层提取最后一个token的表示
    """
    if not hidden_states_tuple:
        return None
    
    last_layer = hidden_states_tuple[-1]
    if not isinstance(last_layer, torch.Tensor):
        return None
    
    # 获取最后一个token的隐藏状态
    # last_layer shape: (batch_size, sequence_length, hidden_dim)
    if len(last_layer.shape) >= 2:
        return last_layer[:, -1, :] if last_layer.shape[0] > 1 else last_layer[-1, :]
    return last_layer

def extract_layer_signal(hidden_states_tuple: tuple, strategy: str, sequence_length: int = None) -> torch.Tensor:
    """
    根据指定策略提取隐藏状态信号
    strategy: '25%', '50%', '75%', 'last_layer', 'eos', 'mean_pooling', 'layer_X'
    """
    strategy = strategy.lower()
    
    if strategy == '25%':
        return get_layer_by_percentage(hidden_states_tuple, 0.25)
    elif strategy == '50%':
        return get_layer_by_percentage(hidden_states_tuple, 0.5)
    elif strategy == '75%':
        return get_layer_by_percentage(hidden_states_tuple, 0.75)
    elif strategy == 'last_layer':
        return get_last_layer(hidden_states_tuple)
    elif strategy == 'eos':
        return get_eos_token_hidden_state(hidden_states_tuple, sequence_length)
    elif strategy == 'mean_pooling':
        return get_mean_pooling(hidden_states_tuple)
    elif strategy.startswith('layer_'):
        try:
            layer_idx = int(strategy.split('_')[1])
            if 0 <= layer_idx < len(hidden_states_tuple):
                return hidden_states_tuple[layer_idx]
            else:
                return None
        except:
            return None
    else:
        raise ValueError(f"Unknown layer extraction strategy: {strategy}")

def compute_layer_statistics(hidden_state: torch.Tensor) -> dict:
    """
    提取隐藏状态的丰富统计特征
    Returns: dict with mean, var, std, max, min, range, skew, kurtosis, norm
    """
    if hidden_state is None or not isinstance(hidden_state, torch.Tensor):
        return {
            'mean': 0.0,
            'var': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0,
            'range': 0.0,
            'skew': 0.0,
            'kurt': 0.0,
            'norm': 0.0
        }
    
    flattened = hidden_state.flatten()
    
    mean_val = flattened.mean().item()
    var_val = flattened.var().item()
    std_val = flattened.std().item()
    max_val = flattened.max().item()
    min_val = flattened.min().item()
    
    return {
        'mean': mean_val,
        'var': var_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'range': max_val - min_val,
        'skew': flattened.skew().item() if len(flattened) > 1 else 0.0,
        'kurt': flattened.kurtosis().item() if len(flattened) > 1 else 0.0,
        'norm': hidden_state.norm(dim=-1).mean().item()
    }
