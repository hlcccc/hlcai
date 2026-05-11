import torch
import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import skew, kurtosis

class EarlyWarningMonitor:
    def __init__(self, entropy_threshold=0.7, variance_threshold=0.5, consecutive_threshold=2):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.consecutive_threshold = consecutive_threshold
        self.consecutive_high_uncertainty = 0
        self.triggered = False
    
    def update(self, entropy=None, variance=None):
        return self.check_early_stop(entropy=entropy, variance=variance)
    
    def check_early_stop(self, entropy=None, variance=None):
        is_high = False
        reason = None
        
        if entropy is not None and entropy > self.entropy_threshold:
            is_high = True
            reason = 'entropy'
        if variance is not None and variance > self.variance_threshold:
            is_high = True
            reason = 'variance'
        
        if is_high:
            self.consecutive_high_uncertainty += 1
            if self.consecutive_high_uncertainty >= self.consecutive_threshold:
                self.triggered = True
                return True, reason
        else:
            self.consecutive_high_uncertainty = 0
        
        return False, None
    
    def reset(self):
        self.consecutive_high_uncertainty = 0
        self.triggered = False

class AdaptiveThreshold:
    def __init__(self, initial_alpha=1.0, learning_rate=0.01):
        self.alpha = initial_alpha
        self.learning_rate = learning_rate
    
    def update(self, reward):
        self.alpha += self.learning_rate * reward
        self.alpha = max(0.1, min(10.0, self.alpha))
        return self.alpha

def compute_entropy(scores):
    return compute_sequence_entropy(scores)

def compute_uncertainty_score(entropy, confidence, early_stop_rate, alpha=0.5):
    normalized_entropy = entropy / np.log(50000)
    return alpha * normalized_entropy + (1 - alpha) * early_stop_rate

def compute_sequence_entropy(scores):
    entropies = []
    for logits in scores:
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        entropies.append(entropy.mean().item())
    return entropies

def compute_max_prob(logits):
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.mean().item()

def compute_layer_wise_variance(hidden_states_tuple: tuple) -> list:
    variances = []
    for layer_idx, hidden in enumerate(hidden_states_tuple):
        if hidden is None:
            var = 0.0
        elif isinstance(hidden, tuple):
            hidden = hidden[0] if len(hidden) > 0 else None
            if hidden is None:
                var = 0.0
            else:
                var = hidden.var(dim=0).mean().item()
        else:
            var = hidden.var(dim=0).mean().item()
        variances.append(var)
    return variances

def compute_generation_confidence(entropy, max_prob):
    return (1 - entropy / np.log(50000)) * max_prob

def get_last_layer(hidden_states_tuple: tuple) -> torch.Tensor:
    if len(hidden_states_tuple) > 0:
        result = hidden_states_tuple[-1]
        while isinstance(result, tuple) and len(result) > 0:
            result = result[-1]
        return result
    return None

def get_mean_pooling(hidden_states_tuple: tuple) -> torch.Tensor:
    valid_layers = []
    for h in hidden_states_tuple:
        if h is None:
            continue
        while isinstance(h, tuple) and len(h) > 0:
            h = h[0]
        if h is not None and isinstance(h, torch.Tensor):
            if h.dim() >= 2:
                h = h[:, -1, :] if h.shape[1] > 0 else h.flatten()
            valid_layers.append(h)
    if len(valid_layers) == 0:
        return None
    try:
        return torch.stack(valid_layers).mean(dim=0)
    except:
        valid_layers = [l.flatten()[:min([x.flatten().shape[0] for x in valid_layers])] for l in valid_layers]
        return torch.stack(valid_layers).mean(dim=0)

def get_layer_by_percentage(hidden_states_tuple: tuple, percentage: float) -> torch.Tensor:
    num_layers = len(hidden_states_tuple)
    if num_layers == 0:
        return None
    layer_idx = int(num_layers * percentage)
    layer_idx = min(layer_idx, num_layers - 1)
    result = hidden_states_tuple[layer_idx]
    while isinstance(result, tuple) and len(result) > 0:
        result = result[0]
    return result

def get_eos_token_hidden_state(hidden_states_tuple: tuple, sequence_length: int = None) -> torch.Tensor:
    if len(hidden_states_tuple) == 0:
        return None
    last_layer = hidden_states_tuple[-1]
    while isinstance(last_layer, tuple) and len(last_layer) > 0:
        last_layer = last_layer[0]
    if last_layer is None:
        return None
    if sequence_length is not None and sequence_length > 0:
        return last_layer[sequence_length - 1]
    return last_layer[-1]

def extract_layer_signal(hidden_states_tuple: tuple, strategy: str, sequence_length: int = None) -> torch.Tensor:
    strategy = strategy.lower()
    
    def unwrap_tuple(obj):
        if isinstance(obj, tuple) and len(obj) > 0:
            return unwrap_tuple(obj[0])
        return obj
    
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
                result = hidden_states_tuple[layer_idx]
                return unwrap_tuple(result)
            else:
                return None
        except:
            return None
    else:
        raise ValueError(f"Unknown layer extraction strategy: {strategy}")

def compute_logdet(hidden_states):
    if hidden_states is None:
        return 0.0
    if hidden_states.ndim > 2:
        hidden_states = hidden_states.mean(dim=1)
    cov = hidden_states.T @ hidden_states
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
    try:
        return torch.logdet(cov).item()
    except:
        return 0.0

def compute_eigen_score(hidden_states):
    if hidden_states is None:
        return 0.0
    if hidden_states.ndim > 2:
        hidden_states = hidden_states.mean(dim=1)
    cov = hidden_states.T @ hidden_states
    try:
        vals = torch.linalg.eigvalsh(cov)
        return vals.abs().mean().item()
    except:
        return 0.0

def compute_layer_statistics(hidden_state: torch.Tensor) -> dict:
    """提取丰富的统计特征"""
    if hidden_state is None or not isinstance(hidden_state, torch.Tensor):
        if isinstance(hidden_state, tuple) and len(hidden_state) > 0:
            hidden_state = hidden_state[0]
            if hidden_state is None or not isinstance(hidden_state, torch.Tensor):
                return {
                    'mean': 0.0, 'var': 0.0, 'std': 0.0, 'max': 0.0,
                    'min': 0.0, 'range': 0.0, 'skew': 0.0, 'kurt': 0.0,
                    'norm': 0.0, 'logdet': 0.0, 'eigen_score': 0.0
                }
        else:
            return {
                'mean': 0.0, 'var': 0.0, 'std': 0.0, 'max': 0.0,
                'min': 0.0, 'range': 0.0, 'skew': 0.0, 'kurt': 0.0,
                'norm': 0.0, 'logdet': 0.0, 'eigen_score': 0.0
            }
    
    flattened = hidden_state.flatten()
    mean_val = flattened.mean().item()
    var_val = flattened.var().item()
    std_val = flattened.std().item()
    max_val = flattened.max().item()
    min_val = flattened.min().item()
    
    if len(flattened) >= 2:
        skew_val = float(skew(flattened.cpu().numpy())) if not torch.isnan(flattened).any() else 0.0
        kurt_val = float(kurtosis(flattened.cpu().numpy())) if not torch.isnan(flattened).any() else 0.0
    else:
        skew_val = 0.0
        kurt_val = 0.0
    
    return {
        'mean': mean_val,
        'var': var_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'range': max_val - min_val,
        'skew': skew_val,
        'kurt': kurt_val,
        'norm': hidden_state.norm(dim=-1).mean().item(),
        'logdet': compute_logdet(hidden_state),
        'eigen_score': compute_eigen_score(hidden_state)
    }

def extract_intermediate_states(model_outputs):
    scores = []
    hidden_states = None
    
    if hasattr(model_outputs, 'scores') and model_outputs.scores is not None:
        scores = [s.detach().cpu() for s in model_outputs.scores]
    
    if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
        hidden_states = tuple(h.detach().cpu() for h in model_outputs.hidden_states)
    
    return scores, hidden_states