"""
LeakIndex 计算模块

用于量化域支表征中的类别语义泄露程度。

核心定义:
    LeakIndex = CDSC - CDDC
    
其中:
- CDSC (Cross-Domain Same-Class): 跨域同类样本相似度
- CDDC (Cross-Domain Different-Class): 跨域异类样本相似度

解释:
- LeakIndex > 0: 域支在按类别聚类 → 语义泄露
- LeakIndex ≈ 0: 域支不携带类别信息 → 理想状态
- LeakIndex < 0: 域支过度抑制同类相似度 → 可能过拟合域轴

参考: leakindex.md
"""

import numpy as np
from typing import Dict, Union


def compute_leak_index(
    cdsc: float,
    cddc: float,
    mode: str = "raw_diff",
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    计算 LeakIndex 和 LeakIntensity
    
    Args:
        cdsc: 跨域同类相似度 (Cross-Domain Same-Class)
        cddc: 跨域异类相似度 (Cross-Domain Different-Class)
        mode: 计算模式
            - "raw_diff": LeakIndex = CDSC - CDDC (推荐，主定义)
            - "norm_gap": LeakIndex = (CDSC - CDDC) / (1 - CDDC + eps)
        eps: 数值稳定性常数（用于 norm_gap 模式）
        
    Returns:
        包含以下键的字典:
        - leak_index: 带符号的泄露指数
        - leak_intensity: 泄露强度（仅正值部分）
        - cdsc: 输入的 CDSC 值（用于记录）
        - cddc: 输入的 CDDC 值（用于记录）
        - mode: 使用的计算模式
        
    Examples:
        >>> compute_leak_index(0.6, 0.3)
        {'leak_index': 0.3, 'leak_intensity': 0.3, 'cdsc': 0.6, 'cddc': 0.3, 'mode': 'raw_diff'}
        
        >>> compute_leak_index(0.3, 0.6)
        {'leak_index': -0.3, 'leak_intensity': 0.0, 'cdsc': 0.3, 'cddc': 0.6, 'mode': 'raw_diff'}
    """
    # 参数验证
    if not isinstance(cdsc, (int, float, np.number)):
        raise TypeError(f"cdsc 必须是数值类型，当前类型: {type(cdsc)}")
    if not isinstance(cddc, (int, float, np.number)):
        raise TypeError(f"cddc 必须是数值类型，当前类型: {type(cddc)}")
    
    # 转换为 float（确保类型一致）
    cdsc = float(cdsc)
    cddc = float(cddc)
    
    # 计算 LeakIndex
    if mode == "raw_diff":
        # 主定义：简单差值
        leak_index = cdsc - cddc
    elif mode == "norm_gap":
        # 归一化间隔（可选）
        leak_index = (cdsc - cddc) / (1.0 - cddc + eps)
    else:
        raise ValueError(f"不支持的计算模式: {mode}，可选值: ['raw_diff', 'norm_gap']")
    
    # 计算 LeakIntensity（仅保留正值部分）
    leak_intensity = max(0.0, leak_index)
    
    return {
        'leak_index': leak_index,
        'leak_intensity': leak_intensity,
        'cdsc': cdsc,
        'cddc': cddc,
        'mode': mode
    }


def compute_leak_index_batch(
    cdsc_list: Union[list, np.ndarray],
    cddc_list: Union[list, np.ndarray],
    mode: str = "raw_diff",
    eps: float = 1e-8
) -> Dict[str, Union[float, np.ndarray]]:
    """
    批量计算 LeakIndex（用于 epoch 级别的统计）
    
    Args:
        cdsc_list: CDSC 值列表（episode 级别）
        cddc_list: CDDC 值列表（episode 级别）
        mode: 计算模式
        eps: 数值稳定性常数
        
    Returns:
        包含以下键的字典:
        - leak_index_mean: LeakIndex 均值
        - leak_index_std: LeakIndex 标准差
        - leak_intensity_mean: LeakIntensity 均值
        - leak_intensity_std: LeakIntensity 标准差
        - num_valid: 有效样本数量
        
    Examples:
        >>> cdsc = [0.6, 0.5, 0.7]
        >>> cddc = [0.3, 0.4, 0.2]
        >>> compute_leak_index_batch(cdsc, cddc)
        {'leak_index_mean': 0.3, 'leak_index_std': 0.082, ...}
    """
    # 转换为 numpy 数组
    cdsc_arr = np.array(cdsc_list, dtype=np.float32)
    cddc_arr = np.array(cddc_list, dtype=np.float32)
    
    # 验证长度一致
    if len(cdsc_arr) != len(cddc_arr):
        raise ValueError(f"cdsc_list 和 cddc_list 长度不一致: {len(cdsc_arr)} vs {len(cddc_arr)}")
    
    # 过滤 NaN 值
    valid_mask = ~(np.isnan(cdsc_arr) | np.isnan(cddc_arr))
    cdsc_valid = cdsc_arr[valid_mask]
    cddc_valid = cddc_arr[valid_mask]
    
    if len(cdsc_valid) == 0:
        # 所有值都是 NaN
        return {
            'leak_index_mean': np.nan,
            'leak_index_std': np.nan,
            'leak_intensity_mean': np.nan,
            'leak_intensity_std': np.nan,
            'num_valid': 0
        }
    
    # 批量计算 LeakIndex
    if mode == "raw_diff":
        leak_index_arr = cdsc_valid - cddc_valid
    elif mode == "norm_gap":
        leak_index_arr = (cdsc_valid - cddc_valid) / (1.0 - cddc_valid + eps)
    else:
        raise ValueError(f"不支持的计算模式: {mode}")
    
    # 计算 LeakIntensity
    leak_intensity_arr = np.maximum(0.0, leak_index_arr)
    
    return {
        'leak_index_mean': float(np.mean(leak_index_arr)),
        'leak_index_std': float(np.std(leak_index_arr)),
        'leak_intensity_mean': float(np.mean(leak_intensity_arr)),
        'leak_intensity_std': float(np.std(leak_intensity_arr)),
        'num_valid': int(len(cdsc_valid))
    }


# 便捷函数：直接从 domain_supcon_stats 计算
def compute_leak_from_stats(
    domain_supcon_stats: Dict[str, float],
    mode: str = "raw_diff",
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    从 domain_supcon_stats 字典直接计算 LeakIndex
    
    Args:
        domain_supcon_stats: DomainSupCon 返回的统计字典
        mode: 计算模式
        eps: 数值稳定性常数
        
    Returns:
        LeakIndex 计算结果字典
        
    Examples:
        >>> stats = {'avg_cdsc_sim': 0.6, 'avg_cddc_sim': 0.3}
        >>> compute_leak_from_stats(stats)
        {'leak_index': 0.3, 'leak_intensity': 0.3, ...}
    """
    # 检查必需的键
    if 'avg_cdsc_sim' not in domain_supcon_stats:
        raise KeyError("domain_supcon_stats 缺少 'avg_cdsc_sim' 键")
    if 'avg_cddc_sim' not in domain_supcon_stats:
        raise KeyError("domain_supcon_stats 缺少 'avg_cddc_sim' 键")
    
    cdsc = domain_supcon_stats['avg_cdsc_sim']
    cddc = domain_supcon_stats['avg_cddc_sim']
    
    return compute_leak_index(cdsc, cddc, mode=mode, eps=eps)
