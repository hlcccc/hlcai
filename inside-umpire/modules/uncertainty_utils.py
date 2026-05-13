import math

import numpy as np
import torch
from scipy.stats import kurtosis, skew


DEFAULT_VOCAB_SIZE = 50000


def _safe_numpy(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _flatten_tensor(hidden_state):
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        for item in hidden_state:
            flattened = _flatten_tensor(item)
            if flattened is not None:
                return flattened
        return None
    if not isinstance(hidden_state, torch.Tensor):
        return None
    if hidden_state.numel() == 0:
        return None
    return hidden_state.detach().float().cpu()


def reduce_to_token_vector(hidden_state):
    hidden_state = _flatten_tensor(hidden_state)
    if hidden_state is None:
        return None
    if hidden_state.dim() == 3:
        return hidden_state[0, -1, :]
    if hidden_state.dim() == 2:
        return hidden_state[-1, :]
    return hidden_state.reshape(-1)


class EarlyWarningMonitor:
    def __init__(
        self,
        entropy_threshold=0.7,
        variance_threshold=0.5,
        consecutive_threshold=2,
        confidence_threshold=0.35,
        drift_threshold=0.25,
        risk_threshold=0.62,
        warmup_steps=2,
        weights=None,
    ):
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.consecutive_threshold = consecutive_threshold
        self.confidence_threshold = confidence_threshold
        self.drift_threshold = drift_threshold
        self.risk_threshold = risk_threshold
        self.warmup_steps = warmup_steps
        self.weights = weights or {
            "entropy": 0.35,
            "confidence_gap": 0.25,
            "layer_spread": 0.20,
            "drift": 0.20,
        }
        self.reset()

    def _normalize_metric(self, value):
        if value is None:
            return 0.0
        value = float(value)
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _dynamic_boost(self, name, value):
        history = self.history[name]
        if value is None or len(history) < self.warmup_steps:
            return 0.0, False
        mean = float(np.mean(history))
        std = float(np.std(history) + 1e-6)
        z_score = (float(value) - mean) / std
        clipped_score = float(np.clip(z_score - 1.0, -60.0, 60.0))
        boost = 1.0 / (1.0 + math.exp(-clipped_score))
        return float(boost), z_score > 2.0

    def _risk_reason(self, metrics):
        dominant = max(metrics.items(), key=lambda item: item[1])[0]
        if dominant == "confidence_gap":
            return "confidence"
        return dominant

    def update(self, entropy=None, variance=None, confidence_gap=None, layer_spread=None, drift=None):
        if layer_spread is None:
            layer_spread = variance

        metrics = {
            "entropy": self._normalize_metric(entropy),
            "confidence_gap": self._normalize_metric(confidence_gap),
            "layer_spread": self._normalize_metric(layer_spread),
            "drift": self._normalize_metric(drift),
        }

        dynamic_flags = []
        boosted_metrics = {}
        for name, value in metrics.items():
            boost, is_dynamic_outlier = self._dynamic_boost(name, value)
            merged_value = 0.5 * value + 0.5 * boost
            boosted_metrics[name] = 0.0 if not np.isfinite(merged_value) else merged_value
            dynamic_flags.append(is_dynamic_outlier)

        risk_score = 0.0
        for name, weight in self.weights.items():
            risk_score += weight * boosted_metrics.get(name, 0.0)
        if not np.isfinite(risk_score):
            risk_score = 0.0

        threshold_trigger = (
            metrics["entropy"] >= self.entropy_threshold
            or metrics["confidence_gap"] >= self.confidence_threshold
            or metrics["layer_spread"] >= self.variance_threshold
            or metrics["drift"] >= self.drift_threshold
        )
        dynamic_trigger = sum(dynamic_flags) >= 2
        is_high = risk_score >= self.risk_threshold or threshold_trigger or dynamic_trigger

        if is_high:
            self.consecutive_high_uncertainty += 1
        else:
            self.consecutive_high_uncertainty = 0

        should_stop = self.consecutive_high_uncertainty >= self.consecutive_threshold
        if should_stop:
            self.triggered = True
            reason = self._risk_reason(boosted_metrics)
        else:
            reason = None

        for name, value in metrics.items():
            self.history[name].append(value)

        snapshot = {
            "risk_score": float(risk_score),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "dynamic_trigger": bool(dynamic_trigger),
            "threshold_trigger": bool(threshold_trigger),
            "triggered": bool(should_stop),
            "reason": reason,
        }
        self.snapshots.append(snapshot)
        return should_stop, reason, snapshot

    def check_early_stop(self, entropy=None, variance=None, confidence_gap=None, layer_spread=None, drift=None):
        return self.update(
            entropy=entropy,
            variance=variance,
            confidence_gap=confidence_gap,
            layer_spread=layer_spread,
            drift=drift,
        )

    def reset(self):
        self.consecutive_high_uncertainty = 0
        self.triggered = False
        self.history = {
            "entropy": [],
            "confidence_gap": [],
            "layer_spread": [],
            "drift": [],
        }
        self.snapshots = []


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


def compute_uncertainty_score(
    entropy,
    confidence,
    early_stop_rate,
    layer_instability=0.0,
    disagreement=0.0,
    alpha=0.35,
):
    confidence_gap = 1.0 - float(confidence)
    return (
        alpha * float(entropy)
        + 0.25 * confidence_gap
        + 0.20 * float(early_stop_rate)
        + 0.10 * float(layer_instability)
        + 0.10 * float(disagreement)
    )


def compute_sequence_entropy(scores):
    entropies = []
    for logits in scores:
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        entropies.append(entropy.mean().item())
    return entropies


def compute_token_entropy_from_logits(logits):
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=1e-12, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return float(entropy.mean().item())


def compute_max_prob(logits):
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return float(probs.max(dim=-1).values.mean().item())


def compute_layer_wise_variance(hidden_states_tuple):
    variances = []
    if hidden_states_tuple is None:
        return variances
    for hidden in hidden_states_tuple:
        hidden = _flatten_tensor(hidden)
        if hidden is None:
            variances.append(0.0)
            continue
        if hidden.dim() == 1:
            variances.append(0.0)
        else:
            variances.append(float(hidden.var(dim=0).mean().item()))
    return variances


def compute_generation_confidence(entropy, max_prob, vocab_size=DEFAULT_VOCAB_SIZE):
    if not np.isfinite(entropy):
        entropy = 0.0
    if not np.isfinite(max_prob):
        max_prob = 0.0
    normalizer = math.log(max(vocab_size, 2))
    normalized_entropy = float(entropy) / max(normalizer, 1e-8)
    return float(np.clip((1.0 - normalized_entropy) * float(max_prob), 0.0, 1.0))


def get_last_layer(hidden_states_tuple):
    if hidden_states_tuple is None or len(hidden_states_tuple) == 0:
        return None
    return _flatten_tensor(hidden_states_tuple[-1])


def get_mean_pooling(hidden_states_tuple):
    if hidden_states_tuple is None:
        return None
    vectors = []
    for hidden in hidden_states_tuple:
        vector = reduce_to_token_vector(hidden)
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def get_layer_by_percentage(hidden_states_tuple, percentage):
    if hidden_states_tuple is None or len(hidden_states_tuple) == 0:
        return None
    layer_idx = int((len(hidden_states_tuple) - 1) * percentage)
    layer_idx = max(0, min(layer_idx, len(hidden_states_tuple) - 1))
    return _flatten_tensor(hidden_states_tuple[layer_idx])


def get_eos_token_hidden_state(hidden_states_tuple, sequence_length=None):
    last_layer = get_last_layer(hidden_states_tuple)
    if last_layer is None:
        return None
    if last_layer.dim() == 3:
        if sequence_length is not None and sequence_length > 0:
            sequence_length = min(sequence_length, last_layer.shape[1])
            return last_layer[0, sequence_length - 1, :]
        return last_layer[0, -1, :]
    if last_layer.dim() == 2:
        if sequence_length is not None and sequence_length > 0:
            sequence_length = min(sequence_length, last_layer.shape[0])
            return last_layer[sequence_length - 1, :]
        return last_layer[-1, :]
    return last_layer


def extract_layer_signal(hidden_states_tuple, strategy, sequence_length=None):
    if hidden_states_tuple is None:
        return None

    strategy = strategy.lower()

    if strategy == "25%":
        return get_layer_by_percentage(hidden_states_tuple, 0.25)
    if strategy == "50%":
        return get_layer_by_percentage(hidden_states_tuple, 0.50)
    if strategy == "75%":
        return get_layer_by_percentage(hidden_states_tuple, 0.75)
    if strategy == "last_layer":
        return get_last_layer(hidden_states_tuple)
    if strategy == "eos":
        return get_eos_token_hidden_state(hidden_states_tuple, sequence_length=sequence_length)
    if strategy == "mean_pooling":
        return get_mean_pooling(hidden_states_tuple)
    if strategy.startswith("layer_"):
        try:
            layer_idx = int(strategy.split("_")[1])
        except (IndexError, ValueError):
            return None
        if 0 <= layer_idx < len(hidden_states_tuple):
            return _flatten_tensor(hidden_states_tuple[layer_idx])
        return None
    raise ValueError(f"Unknown layer extraction strategy: {strategy}")


def compute_logdet(hidden_states):
    hidden_states = _flatten_tensor(hidden_states)
    if hidden_states is None:
        return 0.0
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    cov = hidden_states @ hidden_states.T
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    try:
        return float(torch.logdet(cov).item())
    except Exception:
        return 0.0


def compute_eigen_score(hidden_states):
    hidden_states = _flatten_tensor(hidden_states)
    if hidden_states is None:
        return 0.0
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    cov = hidden_states @ hidden_states.T
    try:
        values = torch.linalg.eigvalsh(cov)
        return float(values.abs().mean().item())
    except Exception:
        return 0.0


def compute_temporal_drift(trajectory):
    trajectory = _flatten_tensor(trajectory)
    if trajectory is None or trajectory.dim() != 2 or trajectory.shape[0] < 2:
        return 0.0
    current = trajectory[1:]
    previous = trajectory[:-1]
    cosine = torch.nn.functional.cosine_similarity(current, previous, dim=-1)
    return float((1.0 - cosine).mean().item())


def compute_delta_norm(trajectory):
    trajectory = _flatten_tensor(trajectory)
    if trajectory is None or trajectory.dim() != 2 or trajectory.shape[0] < 2:
        return 0.0
    delta = trajectory[1:] - trajectory[:-1]
    return float(delta.norm(dim=-1).mean().item())


def compute_layer_spread(layer_vectors):
    valid = []
    for vector in layer_vectors:
        vector = reduce_to_token_vector(vector)
        if vector is not None:
            valid.append(vector)
    if len(valid) < 2:
        return 0.0
    valid = torch.stack(valid, dim=0)
    valid = torch.nn.functional.normalize(valid, dim=-1)
    sim = valid @ valid.T
    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
    return float((1.0 - upper).mean().item())


def compute_layer_statistics(hidden_state):
    hidden_state = _flatten_tensor(hidden_state)
    if hidden_state is None:
        return {
            "mean": 0.0,
            "var": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
            "range": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "norm": 0.0,
            "logdet": 0.0,
            "eigen_score": 0.0,
            "drift": 0.0,
            "delta_norm": 0.0,
        }

    if hidden_state.dim() == 1:
        matrix = hidden_state.unsqueeze(0)
    elif hidden_state.dim() == 2:
        matrix = hidden_state
    else:
        matrix = hidden_state.reshape(hidden_state.shape[0], -1)

    flattened = matrix.flatten()
    np_values = flattened.numpy()

    if np_values.size > 1 and np.std(np_values) > 1e-12:
        skew_val = float(np.nan_to_num(skew(np_values)))
        kurt_val = float(np.nan_to_num(kurtosis(np_values)))
    else:
        skew_val = 0.0
        kurt_val = 0.0

    return {
        "mean": float(flattened.mean().item()),
        "var": float(flattened.var(unbiased=False).item()),
        "std": float(flattened.std(unbiased=False).item()),
        "max": float(flattened.max().item()),
        "min": float(flattened.min().item()),
        "range": float((flattened.max() - flattened.min()).item()),
        "skew": skew_val,
        "kurt": kurt_val,
        "norm": float(matrix.norm(dim=-1).mean().item()),
        "logdet": compute_logdet(matrix),
        "eigen_score": compute_eigen_score(matrix),
        "drift": compute_temporal_drift(matrix),
        "delta_norm": compute_delta_norm(matrix),
    }


def summarize_generation_signals(token_entropies, confidence_scores, layer_spreads, drift_scores, warning_scores):
    token_entropies = [x for x in token_entropies if x is not None and np.isfinite(x)]
    confidence_scores = [x for x in confidence_scores if x is not None and np.isfinite(x)]
    layer_spreads = [x for x in layer_spreads if x is not None and np.isfinite(x)]
    drift_scores = [x for x in drift_scores if x is not None and np.isfinite(x)]
    warning_scores = [x for x in warning_scores if x is not None and np.isfinite(x)]

    if token_entropies:
        weights = np.linspace(1.4, 0.8, num=len(token_entropies))
        entropy_signal = float(np.average(token_entropies, weights=weights))
    else:
        entropy_signal = 0.0

    if confidence_scores:
        weights = np.linspace(1.4, 0.8, num=len(confidence_scores))
        confidence_signal = float(np.average(1.0 - np.asarray(confidence_scores), weights=weights))
    else:
        confidence_signal = 0.0

    if layer_spreads or drift_scores:
        spread_part = float(np.mean(layer_spreads)) if layer_spreads else 0.0
        drift_part = float(np.mean(drift_scores)) if drift_scores else 0.0
        layer_instability = 0.5 * spread_part + 0.5 * drift_part
    else:
        layer_instability = 0.0

    warning_signal = float(np.mean(warning_scores)) if warning_scores else 0.0
    return {
        "entropy_signal": entropy_signal,
        "confidence_signal": confidence_signal,
        "layer_instability": layer_instability,
        "warning_signal": warning_signal,
    }


def extract_intermediate_states(model_outputs):
    scores = []
    hidden_states = None

    if hasattr(model_outputs, "scores") and model_outputs.scores is not None:
        scores = [score.detach().cpu() for score in model_outputs.scores]

    if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
        hidden_states = tuple(_flatten_tensor(hidden) for hidden in model_outputs.hidden_states)

    return scores, hidden_states
