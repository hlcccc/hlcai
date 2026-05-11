import torch
import numpy as np

def compute_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def compute_sequence_entropy(scores: list) -> list:
    entropies = []
    for logits in scores:
        if logits is not None:
            entropies.append(compute_entropy(logits))
        else:
            entropies.append(0.0)
    return entropies

def compute_max_prob(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return probs.max().item()

def compute_generation_confidence(entropy: float, max_prob: float) -> float:
    return max_prob * (1 - entropy)

def compute_layer_wise_variance(hidden_states_tuple: tuple) -> list:
    variances = []
    for layer_idx, hidden in enumerate(hidden_states_tuple):
        if hidden is not None:
            var = hidden.var(dim=0).mean().item()
        else:
            var = 0.0
        variances.append(var)
    return variances

def get_last_layer(hidden_states_tuple: tuple) -> torch.Tensor:
    if len(hidden_states_tuple) > 0:
        return hidden_states_tuple[-1]
    return None

def get_mean_pooling(hidden_states_tuple: tuple) -> torch.Tensor:
    valid_layers = [h for h in hidden_states_tuple if h is not None]
    if len(valid_layers) == 0:
        return None
    return torch.stack(valid_layers).mean(dim=0)

def get_layer_by_percentage(hidden_states_tuple: tuple, percentage: float) -> torch.Tensor:
    num_layers = len(hidden_states_tuple)
    if num_layers == 0:
        return None
    layer_idx = int(num_layers * percentage)
    layer_idx = min(layer_idx, num_layers - 1)
    return hidden_states_tuple[layer_idx]

def get_eos_token_hidden_state(hidden_states_tuple: tuple, sequence_length: int = None) -> torch.Tensor:
    if len(hidden_states_tuple) == 0:
        return None
    last_layer = hidden_states_tuple[-1]
    if last_layer is None:
        return None
    if sequence_length is not None and sequence_length > 0:
        return last_layer[sequence_length - 1]
    return last_layer[-1]

def extract_layer_signal(hidden_states_tuple: tuple, strategy: str, sequence_length: int = None) -> torch.Tensor:
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

def compute_logdet(hidden_states: torch.Tensor, eps: float = 1e-6) -> float:
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    if hidden_states.ndim > 2:
        hidden_states = hidden_states.mean(dim=1)
    
    if hidden_states.ndim == 1:
        hidden_states = hidden_states.unsqueeze(0)
    
    try:
        cov = hidden_states.T @ hidden_states
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
        return torch.logdet(cov).item()
    except Exception as e:
        return 0.0

def compute_eigen_score(hidden_states: torch.Tensor) -> float:
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    if hidden_states.ndim > 2:
        hidden_states = hidden_states.mean(dim=1)
    
    if hidden_states.ndim == 1:
        hidden_states = hidden_states.unsqueeze(0)
    
    try:
        cov = hidden_states.T @ hidden_states
        vals = torch.linalg.eigvalsh(cov)
        return vals.abs().mean().item()
    except Exception as e:
        return 0.0

def compute_layer_statistics(hidden_state: torch.Tensor) -> dict:
    """提取丰富的统计特征"""
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
            'norm': 0.0,
            'logdet': 0.0,
            'eigen_score': 0.0
        }
    
    flattened = hidden_state.flatten()
    np_flattened = flattened.cpu().numpy()
    
    mean_val = flattened.mean().item()
    var_val = flattened.var().item()
    std_val = flattened.std().item()
    max_val = flattened.max().item()
    min_val = flattened.min().item()
    
    if len(np_flattened) > 1:
        from scipy.stats import skew, kurtosis
        skew_val = float(skew(np_flattened))
        kurt_val = float(kurtosis(np_flattened))
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

class EarlyWarningMonitor:
    def __init__(self, entropy_threshold=0.7, variance_threshold=0.5, consecutive_threshold=2):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.consecutive_threshold = consecutive_threshold
        self.consecutive_high_uncertainty = 0
        self.triggered = False
    
    def check_early_stop(self, entropy=None, variance=None):
        is_high = False
        
        if entropy is not None and entropy > self.entropy_threshold:
            is_high = True
        if variance is not None and variance > self.variance_threshold:
            is_high = True
        
        if is_high:
            self.consecutive_high_uncertainty += 1
            if self.consecutive_high_uncertainty >= self.consecutive_threshold:
                self.triggered = True
                return True
        else:
            self.consecutive_high_uncertainty = 0
        
        return False
    
    def reset(self):
        self.consecutive_high_uncertainty = 0
        self.triggered = False

class AdaptiveThreshold:
    def __init__(self, initial_alpha=1.0, learning_rate=0.01):
        self.alpha = initial_alpha
        self.learning_rate = learning_rate
    
    def update(self, error_rate):
        self.alpha = self.alpha + self.learning_rate * (error_rate - 0.5)
        self.alpha = max(0.1, min(2.0, self.alpha))
        return self.alpha

def compute_uncertainty_score(entropy: float, confidence: float, early_stop_rate: float, 
                              alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> float:
    return alpha * entropy + beta * (1 - confidence) + gamma * early_stop_rate

def compute_early_stop_indicator(uncertainty_info: dict) -> float:
    if not uncertainty_info:
        return 0.0
    return uncertainty_info.get('early_stop_triggered', 0.0)

def compute_avg_token_entropy(uncertainty_info: dict) -> float:
    if not uncertainty_info:
        return 0.0
    entropies = uncertainty_info.get('token_entropies', [])
    if len(entropies) == 0:
        return 0.0
    return np.mean(entropies)

def compute_avg_confidence(uncertainty_info: dict) -> float:
    if not uncertainty_info:
        return 0.0
    confidence_scores = uncertainty_info.get('confidence_scores', [])
    if len(confidence_scores) == 0:
        return 0.0
    return np.mean(confidence_scores)

def compute_generation_diversity(generations: list) -> float:
    if len(generations) < 2:
        return 0.0
    
    unique_gens = len(set(generations))
    return unique_gens / len(generations)
