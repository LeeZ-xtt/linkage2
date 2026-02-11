import torch
import numpy as np
from typing import List, Tuple


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算Top-1准确率
    
    Args:
        predictions: 模型预测结果，形状为[N, num_classes]
        targets: 真实标签，形状为[N]
        
    Returns:
        float: Top-1准确率
    """
    with torch.no_grad():
        predicted_labels = torch.argmax(predictions, dim=1)
        correct = (predicted_labels == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
    return accuracy


def compute_confidence_interval(accuracies: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    计算准确率的置信区间
    
    Args:
        accuracies: 多次测量的准确率列表
        confidence_level: 置信水平，默认为0.95
        
    Returns:
        Tuple[float, float]: 置信区间的下限和上限
    """
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # 计算标准误差
    n = len(accuracies)
    standard_error = std_acc / np.sqrt(n)
    
    # 计算置信区间的临界值 (使用正态分布近似)
    z_value = 1.96 if confidence_level == 0.95 else 2.576  # 0.95对应1.96, 0.99对应2.576
    
    # 计算置信区间
    margin_error = z_value * standard_error
    lower_bound = mean_acc - margin_error
    upper_bound = mean_acc + margin_error
    
    return lower_bound, upper_bound


def compute_episode_accuracy(query_logits: torch.Tensor, query_labels: torch.Tensor) -> float:
    """
    计算单个episode的准确率
    
    Args:
        query_logits: 查询集的logits，形状为[num_query, num_classes]
        query_labels: 查询集的真实标签，形状为[num_query]
        
    Returns:
        float: episode准确率
    """
    return compute_accuracy(query_logits, query_labels)


def compute_epoch_accuracy(episode_accuracies: List[float]) -> float:
    """
    计算epoch的平均准确率
    
    Args:
        episode_accuracies: 一个epoch中所有episode的准确率列表
        
    Returns:
        float: epoch平均准确率
    """
    if not episode_accuracies:
        return 0.0
    return np.mean(episode_accuracies)


def compute_epoch_statistics(episode_accuracies: List[float]) -> Tuple[float, float, float]:
    """
    计算epoch的统计信息
    
    Args:
        episode_accuracies: 一个epoch中所有episode的准确率列表
        
    Returns:
        Tuple[float, float, float]: (平均准确率, 标准差, 标准误差)
    """
    if not episode_accuracies:
        return 0.0, 0.0, 0.0
    
    accuracies = np.array(episode_accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    std_error = std_acc / np.sqrt(len(accuracies))
    
    return mean_acc, std_acc, std_error


def track_epoch_progress(epoch_accuracies: List[List[float]]) -> List[float]:
    """
    跟踪多个epoch的准确率进展
    
    Args:
        epoch_accuracies: 每个epoch的准确率列表的列表
        
    Returns:
        List[float]: 每个epoch的平均准确率列表
    """
    epoch_means = []
    for episode_accs in epoch_accuracies:
        epoch_means.append(compute_epoch_accuracy(episode_accs))
    return epoch_means


def compute_prototype_separation_ratio(support_features: torch.Tensor, support_labels: torch.Tensor, prototypes: torch.Tensor, block_size: int = 1024) -> dict:
    """
    计算原型分离度（类间距离与类内平均距离之比）
    
    Args:
        support_features: 支持集特征，形状为[N, feature_dim]
        support_labels: 支持集真实标签，形状为[N]
        prototypes: 各类原型向量，形状为[num_classes, feature_dim]
        block_size: 分块处理大小，用于节省显存，默认为1024
        
    Returns:
        dict: 包含类间距离(d_inter)、类内平均距离(d_intra)与分离度(separation_ratio)的字典
    """
    device = prototypes.device
    support_features = support_features.to(device).float()
    support_labels = support_labels.to(device)
    prototypes = prototypes.to(device).float()

    uniq, inv = torch.unique(support_labels, sorted=True, return_inverse=True)
    n_way = prototypes.size(0)

    if n_way > 1:
        pd = torch.pdist(prototypes, p=2)
        d_inter = pd.sum() * (2.0 / (n_way * (n_way - 1)))
    else:
        d_inter = torch.tensor(0.0, device=device)

    if support_features.size(0) > block_size:
        sums = torch.zeros(n_way, device=device, dtype=torch.float32)
        counts = torch.zeros(n_way, device=device, dtype=torch.float32)
        for start in range(0, support_features.size(0), block_size):
            end = min(start + block_size, support_features.size(0))
            sf_blk = support_features[start:end]
            inv_blk = inv[start:end]
            centers_blk = prototypes.index_select(0, inv_blk)
            dists_blk = torch.norm(sf_blk - centers_blk, dim=1)
            sums.index_add_(0, inv_blk, dists_blk)
            counts.index_add_(0, inv_blk, torch.ones_like(dists_blk))
    else:
        centers = prototypes.index_select(0, inv)
        dists = torch.norm(support_features - centers, dim=1)
        sums = torch.bincount(inv, weights=dists, minlength=n_way).float()
        counts = torch.bincount(inv, minlength=n_way).float()

    valid_mask = counts > 1
    if valid_mask.any():
        class_means = sums[valid_mask] / counts[valid_mask]
        d_intra = class_means.mean()
    else:
        print('[SeparationRatio] intra-class distance is zero or undefined')
        return {'d_inter': float(d_inter.item()), 'd_intra': 0.0, 'separation_ratio': float('inf')}

    ratio = d_inter / d_intra if d_intra > 0 else torch.tensor(float('inf'), device=device)

    # 内部工具函数：将张量安全地转换为 float，若值为无穷或非数则返回 -1.0
    def _sanitize(x: torch.Tensor) -> float:
        return float(x.item()) if torch.isfinite(x) else -1.0

    # 返回包含三个关键指标的字典：
    # d_inter: 类间平均距离
    # d_intra: 类内平均距离
    # separation_ratio: 分离度（类间距离与类内距离之比，越大越好）
    return {
        'd_inter': _sanitize(d_inter),
        'd_intra': _sanitize(d_intra),
        'separation_ratio': _sanitize(ratio),
    }