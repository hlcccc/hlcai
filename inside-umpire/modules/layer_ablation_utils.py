import torch
import numpy as np

def compute_logdet(hidden_states: torch.Tensor, alpha: float = 1e-6) -> float:
    """
    计算协方差矩阵的 logdet（INSIDE 方法）
    量化句子 embedding 的体积
    """
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    # 如果是 3D tensor，先对序列维度取平均
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.mean(dim=1)
    
    # 确保是 2D
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    
    try:
        # 计算协方差矩阵
        cov_matrix = hidden_states @ hidden_states.T
        
        # 添加正则化防止奇异
        cov_matrix = cov_matrix + alpha * torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        
        logdet = torch.logdet(cov_matrix).item()
        return logdet
    except Exception as e:
        return 0.0

def compute_eigen_score(hidden_states: torch.Tensor) -> float:
    """
    计算特征值分数（INSIDE EigenScore）
    量化 embedding 的多样性/发散度
    """
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    # 如果是 3D tensor，先对序列维度取平均
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.mean(dim=1)
    
    # 确保是 2D
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    
    try:
        # 计算协方差矩阵
        cov_matrix = hidden_states @ hidden_states.T
        
        # 计算特征值
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        
        # 返回特征值的平均值
        return eigenvalues.abs().mean().item()
    except Exception as e:
        return 0.0

def compute_incoherence_score(hidden_states: torch.Tensor) -> float:
    """
    计算不一致性分数（UMPIRE 方法）
    衡量样本之间的不一致程度
    """
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    # 如果是 3D tensor，先对序列维度取平均
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.mean(dim=1)
    
    # 确保是 2D
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)
    
    try:
        # 归一化
        norms = hidden_states.norm(dim=-1)
        normalized = hidden_states / norms.unsqueeze(-1).clamp(min=1e-8)
        
        # 计算相似度矩阵
        similarities = normalized @ normalized.T
        
        # 不一致性 = 1 - 平均相似度
        avg_similarity = similarities.mean().item()
        return 1 - avg_similarity
    except Exception as e:
        return 0.0

def compute_spectral_norm(hidden_states: torch.Tensor) -> float:
    """
    计算谱范数（最大奇异值）
    衡量表示的"能量"
    """
    if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
        return 0.0
    
    try:
        return torch.linalg.norm(hidden_states, ord=2).item()
    except Exception as e:
        return 0.0

def compute_layer_ablation_features(hidden_state: torch.Tensor) -> dict:
    """
    综合提取层消融实验所需的所有特征
    包括：统计特征、EigenScore、logdet、不一致性分数
    """
    features = {}
    
    # 统计特征
    if hidden_state is not None and isinstance(hidden_state, torch.Tensor):
        flattened = hidden_state.flatten()
        np_flattened = flattened.cpu().numpy()
        
        mean_val = flattened.mean().item()
        var_val = flattened.var().item()
        std_val = flattened.std().item()
        max_val = flattened.max().item()
        min_val = flattened.min().item()
        
        # 使用 numpy 计算偏度和峰度
        if len(np_flattened) > 1:
            from scipy.stats import skew, kurtosis
            skew_val = float(skew(np_flattened))
            kurt_val = float(kurtosis(np_flattened))
        else:
            skew_val = 0.0
            kurt_val = 0.0
        
        features.update({
            'mean': mean_val,
            'var': var_val,
            'std': std_val,
            'max': max_val,
            'min': min_val,
            'range': max_val - min_val,
            'skew': skew_val,
            'kurt': kurt_val,
            'norm': hidden_state.norm(dim=-1).mean().item(),
            'spectral_norm': compute_spectral_norm(hidden_state),
            'eigen_score': compute_eigen_score(hidden_state),
            'logdet': compute_logdet(hidden_state),
            'incoherence': compute_incoherence_score(hidden_state)
        })
    else:
        features.update({
            'mean': 0.0,
            'var': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0,
            'range': 0.0,
            'skew': 0.0,
            'kurt': 0.0,
            'norm': 0.0,
            'spectral_norm': 0.0,
            'eigen_score': 0.0,
            'logdet': 0.0,
            'incoherence': 0.0
        })
    
    return features
