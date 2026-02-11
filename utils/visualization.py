"""
可视化工具模块
"""

import os
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
from typing import Optional, List, Dict, Tuple, Union

_cjk_candidates = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'PingFang SC',
    'WenQuanYi Zen Hei',
    'Arial Unicode MS'
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_selected_font = next((n for n in _cjk_candidates if n in _available_fonts), 'DejaVu Sans')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [_selected_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _is_interactive_backend() -> bool:
    try:
        backend = matplotlib.get_backend()
    except Exception:
        return False
    interactive_backends = {"TkAgg", "Qt5Agg", "QtAgg", "WebAgg", "MacOSX", "nbAgg"}
    return backend in interactive_backends


def _safe_filename(title: str) -> str:
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in title.strip())
    return name or "figure"


def _default_save_path(title: str) -> str:
    out_dir = os.path.join(".", "figures")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_safe_filename(title)}.png")


def _finalize_figure(fig, title: str, save_path: str | None = None) -> None:
    if save_path is None and not _is_interactive_backend():
        save_path = _default_save_path(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if _is_interactive_backend() and save_path is None:
        plt.show()
    plt.close(fig)


def visualize_alpha_weights(alpha_weights, layer_names, title="Layer Weights Visualization", save_path=None):
    """
    可视化层权重α
    
    Args:
        alpha_weights: 权重数组
        layer_names: 层名称列表
        title: 图表标题
    """
    # 如果alpha_weights是torch.Tensor，则转换为numpy数组
    if isinstance(alpha_weights, torch.Tensor):
        alpha_weights = alpha_weights.detach().cpu().numpy()
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(layer_names, alpha_weights)
    
    # 设置图表属性
    ax.set_xlabel('Layers')
    ax.set_ylabel('Weight Values')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    
    # 在每个条形上显示数值
    for bar, weight in zip(bars, alpha_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_training_curve(train_losses, train_accuracies, val_accuracies=None, title="Training Curve", val_epochs=None, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表（可选）
        title: 图表标题
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个y轴用于准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_accuracies, color=color, label='Train Accuracy')
    
    # 绘制验证准确率（如果提供）
    if val_accuracies is not None and len(val_accuracies) > 0:
        val_x = val_epochs if val_epochs is not None else list(range(10, len(train_losses) + 1, 10))
        ax2.plot(val_x, val_accuracies, color='tab:green', label='Val Accuracy')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title)
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_epoch_accuracy(epoch_accuracies, title="Epoch Average Accuracy", save_path=None):
    """
    绘制epoch平均准确率曲线
    
    Args:
        epoch_accuracies: epoch平均准确率列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(epoch_accuracies) + 1)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_accuracies, 'b-', linewidth=2, marker='o', markersize=6, label='Epoch Accuracy')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    # 添加数值标注（每5个epoch标注一次）
    for i in range(0, len(epoch_accuracies), 5):
        plt.annotate(f'{epoch_accuracies[i]:.3f}', 
                    (epochs[i], epoch_accuracies[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_epoch_statistics(epoch_means, epoch_stds=None, title="Epoch Statistics", save_path=None):
    """
    绘制epoch统计信息（平均值和标准差）
    
    Args:
        epoch_means: epoch平均准确率列表
        epoch_stds: epoch标准差列表（可选）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(epoch_means) + 1)
    
    fig = plt.figure(figsize=(12, 6))
    
    # 绘制平均准确率
    plt.plot(epochs, epoch_means, 'b-', linewidth=2, marker='o', markersize=6, label='Mean Accuracy')
    
    # 如果提供了标准差，绘制误差带
    if epoch_stds is not None:
        epoch_means_arr = np.array(epoch_means)
        epoch_stds_arr = np.array(epoch_stds)
        plt.fill_between(epochs, 
                        epoch_means_arr - epoch_stds_arr, 
                        epoch_means_arr + epoch_stds_arr, 
                        alpha=0.2, color='blue', label='±1 Std Dev')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_accuracy_comparison(train_accuracies, val_accuracies=None, test_accuracies=None, 
                           title="Accuracy Comparison", save_path=None, val_epochs=None, test_epochs=None):
    """
    比较训练、验证和测试准确率
    
    Args:
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表（可选）
        test_accuracies: 测试准确率列表（可选）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    epochs = range(1, len(train_accuracies) + 1)
    
    fig = plt.figure(figsize=(12, 6))
    
    # 绘制训练准确率
    plt.plot(epochs, train_accuracies, 'b-', linewidth=2, marker='o', 
             markersize=4, label='Train Accuracy')
    
    # 绘制验证准确率
    if val_accuracies is not None:
        if val_epochs is None:
            if len(val_accuracies) == len(train_accuracies):
                val_epochs = range(1, len(val_accuracies) + 1)
            else:
                val_epochs = list(range(10, len(train_accuracies) + 1, 10))
        plt.plot(val_epochs, val_accuracies, 'g-', linewidth=2, marker='s', 
                 markersize=4, label='Validation Accuracy')
    
    # 绘制测试准确率
    if test_accuracies is not None:
        if test_epochs is None:
            test_epochs = range(1, len(test_accuracies) + 1)
        plt.plot(test_epochs, test_accuracies, 'r-', linewidth=2, marker='^', 
                 markersize=4, label='Test Accuracy')
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    # 设置y轴范围
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_accuracy_heatmap(accuracy_matrix, class_names=None, metric_names=None, title="Accuracy Heatmap", save_path=None):
    """
    绘制准确率热力图（用于分析不同类别或条件下的准确率）
    
    Args:
        accuracy_matrix: 准确率矩阵，形状为[n_conditions, n_metrics]
        class_names: 类别名称列表（可选，用作行标签）
        metric_names: 指标名称列表（可选，用作列标签）
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 检测数据类型并设置合适的颜色映射和范围
    matrix = np.array(accuracy_matrix)
    
    # 如果数据包含标准差（通常在第三列），需要特殊处理
    if matrix.shape[1] >= 3:
        # 分别处理准确率列（0,1）和标准差列（2+）
        acc_data = matrix[:, :2]  # 准确率数据
        std_data = matrix[:, 2:]  # 标准差数据
        
        # 标准化处理：将标准差缩放到0-1范围以便可视化
        if std_data.size > 0:
            std_normalized = (std_data - std_data.min()) / (std_data.max() - std_data.min() + 1e-8)
            # 重新组合数据
            display_matrix = np.concatenate([acc_data, std_normalized], axis=1)
        else:
            display_matrix = acc_data
    else:
        display_matrix = matrix
    
    # 创建热力图
    im = plt.imshow(display_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # 设置行标签（条件/类别）
    if class_names is not None:
        plt.yticks(range(len(class_names)), class_names, fontproperties=FontProperties(family=_selected_font))
    
    # 设置列标签（指标）
    if metric_names is not None:
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right', fontproperties=FontProperties(family=_selected_font))
    else:
        plt.xlabel('Metrics')
    
    plt.ylabel('Epochs/Conditions')
    plt.title(title)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    if matrix.shape[1] >= 3:
        cbar.set_label('Normalized Values (Accuracy: 0-1, Std: normalized)')
    else:
        cbar.set_label('Accuracy')
    
    # 在每个格子中显示原始数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # 显示原始值，不是标准化后的值
            value = matrix[i, j]
            if j < 2:  # 准确率列
                text_str = f'{value:.3f}'
            else:  # 标准差列
                text_str = f'{value:.4f}'
            
            # 根据背景颜色选择文字颜色
            text_color = "white" if display_matrix[i, j] < 0.5 else "black"
            plt.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=9)
    
    plt.tight_layout()
    
    _finalize_figure(fig, title, save_path)


def plot_val_accuracy_curve(val_accuracies, title="Validation Accuracy Curve", val_epochs=None, save_path=None):
    if isinstance(val_accuracies, torch.Tensor):
        val_accuracies = val_accuracies.detach().cpu().numpy()
    if val_epochs is None:
        val_epochs = range(1, len(val_accuracies) + 1)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_accuracies, 'g-', linewidth=2, marker='s', markersize=6, label='Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


def plot_separation_ratio_curve(sep_ratios, sep_epochs=None, title="Separation Ratio Curve", save_path=None):
    if isinstance(sep_ratios, torch.Tensor):
        sep_ratios = sep_ratios.detach().cpu().numpy()
    if sep_epochs is None:
        sep_epochs = list(range(5, 5 * len(sep_ratios) + 1, 5))
    fig = plt.figure(figsize=(10, 6))
    plt.plot(sep_epochs, sep_ratios, 'm-', linewidth=2, marker='d', markersize=6, label='Separation Ratio')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Separation Ratio')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _finalize_figure(fig, title, save_path)


# =============================================================================
# T-SNE 可视化模块
# =============================================================================

def plot_tsne_embeddings(
    features: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    epoch: int,
    n_way: int = 5,
    domain_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prototypes: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    save_dir: str = "figures/T-SNE",
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
    class_names: Optional[List[str]] = None,
    domain_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    init_embedding: Optional[Union[torch.Tensor, np.ndarray]] = None,
    pca_dim: int = 50,
    high_dim_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    使用 T-SNE 可视化特征嵌入的聚类程度
    
    遵循 sklearn 1.7+ 最佳实践:
    - 高维特征先用 PCA 降至 pca_dim 维（官方推荐）
    - 使用 init='pca' 提高稳定性
    - 使用 learning_rate='auto' 自适应学习率
    
    Args:
        features: 特征矩阵 ``[N, D]``，N为样本数，D为特征维度
        labels: 类别标签 ``[N]``，范围 ``[0, n_way-1]``
        epoch: 当前训练轮数（用于文件命名）
        n_way: 类别数量
        domain_labels: 域标签 ``[N]``（可选，用于区分不同域）
        prototypes: 原型向量 ``[n_way, D]``（可选，将原型投影到T-SNE空间）
        title: 图表标题（可选）
        save_dir: 保存目录
        perplexity: T-SNE困惑度参数，建议值5-50，样本少时用较小值
        max_iter: T-SNE最大迭代次数 (sklearn 1.5+ 使用 max_iter)
        random_state: 随机种子
        class_names: 类别名称列表（可选）
        domain_names: 域名称列表（可选）
        figsize: 图像尺寸
        init_embedding: 上一次的嵌入坐标 [N, 2]，用于保持视觉一致性 (Warm Start)
        pca_dim: PCA预降维目标维度（sklearn官方建议50维）
        
    Returns:
        metrics: 包含聚类度量指标的字典
            - silhouette_score: 轮廓系数 [-1, 1]，越大越好
            - intra_class_dist: 类内平均距离，越小越好
            - inter_class_dist: 类间平均距离，越大越好
            - cluster_ratio: inter/intra比值，越大聚类效果越好
    """
    from sklearn.decomposition import PCA
    
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if domain_labels is not None and isinstance(domain_labels, torch.Tensor):
        domain_labels = domain_labels.detach().cpu().numpy()
    if prototypes is not None and isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().cpu().numpy()
    if init_embedding is not None and isinstance(init_embedding, torch.Tensor):
        init_embedding = init_embedding.detach().cpu().numpy()
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 合并特征和原型（如果提供）
    if prototypes is not None:
        all_features = np.vstack([features, prototypes])
        n_samples = features.shape[0]
    else:
        all_features = features
        n_samples = features.shape[0]
    
    # ===== sklearn官方最佳实践：高维特征先用PCA降维 =====
    # "It is highly recommended to use another dimensionality reduction method 
    #  (e.g. PCA for dense data) to reduce the number of dimensions to a 
    #  reasonable amount (e.g. 50) if the number of features is very high."
    n_features = all_features.shape[1]
    if n_features > pca_dim:
        # 确保PCA组件数不超过样本数和特征数
        n_pca_components = min(pca_dim, all_features.shape[0] - 1, n_features)
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        all_features = pca.fit_transform(all_features)
    
    effective_perplexity = min(perplexity, (all_features.shape[0] - 1) / 3)
    effective_perplexity = max(5.0, effective_perplexity)  # 最小值为5
    
    # 执行T-SNE降维 (sklearn 1.7+ API)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=max_iter,          # sklearn 1.5+ 参数名
        random_state=random_state,   # 用户指定固定随机种子
        init='pca',                  # 用户强制指定 init='pca'
        learning_rate='auto',        # sklearn 1.2+ 默认值，自适应
        n_iter_without_progress=300, # 早停参数
        method='barnes_hut',         # O(NlogN) 复杂度，适合大数据
        angle=0.5,                   # Barnes-Hut 精度-速度权衡
    )
    embeddings_2d = tsne.fit_transform(all_features)
    
    # 分离样本嵌入和原型嵌入
    sample_embeddings = embeddings_2d[:n_samples]
    proto_embeddings = embeddings_2d[n_samples:] if prototypes is not None else None
    
    # 计算聚类度量指标
    # 若提供了高维空间上的预计算指标，则直接使用；否则退回到在2D T-SNE空间上估计
    if high_dim_metrics is not None:
        metrics = high_dim_metrics
    else:
        metrics = _compute_clustering_metrics(sample_embeddings, labels, n_way)
    
    # 设置颜色映射
    cmap = plt.cm.get_cmap('tab10', n_way)
    colors = [cmap(i) for i in range(n_way)]
    
    # 创建图形
    if domain_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))
        ax_class, ax_domain = axes
    else:
        fig, ax_class = plt.subplots(1, 1, figsize=figsize)
        ax_domain = None
    
    # === 按类别着色的散点图 ===
    for c in range(n_way):
        mask = labels == c
        label_name = class_names[c] if class_names else f"类别 {c}"
        ax_class.scatter(
            sample_embeddings[mask, 0],
            sample_embeddings[mask, 1],
            c=[colors[c]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
    
    # 绘制原型（如果提供）
    if proto_embeddings is not None:
        for c in range(n_way):
            ax_class.scatter(
                proto_embeddings[c, 0],
                proto_embeddings[c, 1],
                c=[colors[c]],
                marker='*',
                s=400,
                edgecolors='black',
                linewidths=2,
                zorder=10,
                label=f"原型 {c}" if c == 0 else None
            )
            # 添加原型标注
            ax_class.annotate(
                f'P{c}',
                (proto_embeddings[c, 0], proto_embeddings[c, 1]),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom',
                xytext=(0, 8),
                textcoords='offset points'
            )
    
    ax_class.set_xlabel('T-SNE 维度 1', fontsize=12)
    ax_class.set_ylabel('T-SNE 维度 2', fontsize=12)
    ax_class.legend(loc='best', fontsize=9, framealpha=0.9)
    ax_class.grid(True, alpha=0.3)
    ax_class.set_title(f'按类别着色 (Epoch {epoch})', fontsize=14)
    
    # 添加聚类指标文本框
    textstr = '\n'.join([
        f'轮廓系数: {metrics["silhouette_score"]:.3f}',
        f'类内距离: {metrics["intra_class_dist"]:.3f}',
        f'类间距离: {metrics["inter_class_dist"]:.3f}',
        f'聚类比: {metrics["cluster_ratio"]:.3f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_class.text(
        0.02, 0.98, textstr,
        transform=ax_class.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
        fontproperties=FontProperties(family=_selected_font)
    )
    
    # === 按域着色的散点图（如果提供域标签）===
    if ax_domain is not None and domain_labels is not None:
        unique_domains = np.unique(domain_labels)
        domain_cmap = plt.cm.get_cmap('Set2', len(unique_domains))
        
        # 定义域的标记样式
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        for i, d in enumerate(unique_domains):
            mask = domain_labels == d
            domain_name = domain_names[d] if domain_names and d < len(domain_names) else f"域 {d}"
            ax_domain.scatter(
                sample_embeddings[mask, 0],
                sample_embeddings[mask, 1],
                c=[domain_cmap(i)],
                marker=markers[i % len(markers)],
                label=domain_name,
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )
        
        ax_domain.set_xlabel('T-SNE 维度 1', fontsize=12)
        ax_domain.set_ylabel('T-SNE 维度 2', fontsize=12)
        ax_domain.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_domain.grid(True, alpha=0.3)
        ax_domain.set_title(f'按域着色 (Epoch {epoch})', fontsize=14)
    
    # 设置总标题
    if title is None:
        title = f"T-SNE 特征可视化 - Epoch {epoch}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f"tsne_epoch_{epoch:03d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  📊 T-SNE可视化已保存: {save_path}")
    print(f"     聚类指标 - 轮廓系数: {metrics['silhouette_score']:.4f}, 聚类比: {metrics['cluster_ratio']:.4f}")
    
    return metrics, embeddings_2d


def _compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_way: int
) -> Dict[str, float]:
    """
    计算聚类质量度量指标
    
    Args:
        embeddings: T-SNE降维后的2D坐标 ``[N, 2]``
        labels: 类别标签 ``[N]``
        n_way: 类别数量
        
    Returns:
        metrics: 聚类度量指标字典
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    
    metrics = {}
    
    # 1. 轮廓系数 (Silhouette Score)
    try:
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = sklearn_silhouette(embeddings, labels)
        else:
            metrics['silhouette_score'] = 0.0
    except Exception:
        metrics['silhouette_score'] = 0.0
    
    # 2. 类内平均距离 (Intra-class distance)
    intra_dists = []
    class_centers = []
    for c in range(n_way):
        mask = labels == c
        if mask.sum() > 1:
            class_points = embeddings[mask]
            center = class_points.mean(axis=0)
            class_centers.append(center)
            dists = np.sqrt(((class_points - center) ** 2).sum(axis=1))
            intra_dists.append(dists.mean())
        elif mask.sum() == 1:
            class_centers.append(embeddings[mask][0])
            intra_dists.append(0.0)
    
    metrics['intra_class_dist'] = np.mean(intra_dists) if intra_dists else 0.0
    
    # 3. 类间平均距离 (Inter-class distance)
    if len(class_centers) > 1:
        class_centers = np.array(class_centers)
        inter_dists = []
        for i in range(len(class_centers)):
            for j in range(i + 1, len(class_centers)):
                dist = np.sqrt(((class_centers[i] - class_centers[j]) ** 2).sum())
                inter_dists.append(dist)
        metrics['inter_class_dist'] = np.mean(inter_dists)
    else:
        metrics['inter_class_dist'] = 0.0
    
    # 4. 聚类比 (Cluster Ratio = inter/intra)
    if metrics['intra_class_dist'] > 1e-6:
        metrics['cluster_ratio'] = metrics['inter_class_dist'] / metrics['intra_class_dist']
    else:
        metrics['cluster_ratio'] = float('inf') if metrics['inter_class_dist'] > 0 else 0.0
    
    return metrics


def plot_tsne_evolution(
    metrics_history: List[Dict[str, float]],
    epochs: List[int],
    save_dir: str = "figures/T-SNE",
    title: str = "T-SNE 聚类指标演化"
) -> None:
    """
    绘制T-SNE聚类指标随训练的演化曲线
    
    Args:
        metrics_history: 每个epoch的聚类指标列表
        epochs: 对应的epoch列表
        save_dir: 保存目录
        title: 图表标题
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取各指标
    silhouette = [m['silhouette_score'] for m in metrics_history]
    intra_dist = [m['intra_class_dist'] for m in metrics_history]
    inter_dist = [m['inter_class_dist'] for m in metrics_history]
    cluster_ratio = [m['cluster_ratio'] for m in metrics_history]
    
    # 轮廓系数
    axes[0, 0].plot(epochs, silhouette, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('轮廓系数')
    axes[0, 0].set_title('轮廓系数 (越大越好)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 类内距离
    axes[0, 1].plot(epochs, intra_dist, 'r-s', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('类内平均距离')
    axes[0, 1].set_title('类内距离 (越小越好)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 类间距离
    axes[1, 0].plot(epochs, inter_dist, 'g-^', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('类间平均距离')
    axes[1, 0].set_title('类间距离 (越大越好)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 聚类比
    axes[1, 1].plot(epochs, cluster_ratio, 'm-d', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('聚类比 (Inter/Intra)')
    axes[1, 1].set_title('聚类比 (越大越好)')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "tsne_metrics_evolution.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  📈 T-SNE聚类指标演化图已保存: {save_path}")


def plot_leakindex_curve(
    leakindex_history: List[float],
    leakintensity_history: List[float],
    epochs: Optional[List[int]] = None,
    title: str = "LeakIndex 演化曲线",
    save_path: Optional[str] = None
) -> None:
    """
    绘制 LeakIndex 和 LeakIntensity 演化曲线（单独 subplot）
    
    LeakIndex = CDSC - CDDC (带符号)
    - LeakIndex > 0: 域支在按类别聚类 → 语义泄露
    - LeakIndex ≈ 0: 域支不携带类别信息 → 理想状态
    - LeakIndex < 0: 域支过度抑制同类相似度
    
    LeakIntensity = max(0, LeakIndex) (仅正值部分)
    - 用于监控泄露强度的绝对值
    
    Args:
        leakindex_history: LeakIndex 历史记录（带符号）
        leakintensity_history: LeakIntensity 历史记录（仅正值）
        epochs: epoch 列表（可选，默认从 1 开始）
        title: 图表标题
        save_path: 保存路径（可选）
        
    Examples:
        >>> plot_leakindex_curve([0.1, 0.2, -0.1], [0.1, 0.2, 0.0])
    """
    # 转换为 numpy 数组
    if isinstance(leakindex_history, torch.Tensor):
        leakindex_history = leakindex_history.detach().cpu().numpy()
    if isinstance(leakintensity_history, torch.Tensor):
        leakintensity_history = leakintensity_history.detach().cpu().numpy()
    
    leakindex_arr = np.array(leakindex_history)
    leakintensity_arr = np.array(leakintensity_history)
    
    # 生成 epoch 列表
    if epochs is None:
        epochs = list(range(1, len(leakindex_arr) + 1))
    
    # 创建 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === 左图: LeakIndex (带符号) ===
    ax_leak = axes[0]
    ax_leak.plot(epochs, leakindex_arr, 'b-o', linewidth=2, markersize=6, label='LeakIndex')
    ax_leak.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='理想值 (LeakIndex=0)')
    ax_leak.fill_between(epochs, 0, leakindex_arr, where=(leakindex_arr > 0), 
                         color='red', alpha=0.2, label='泄露区域 (>0)')
    ax_leak.fill_between(epochs, 0, leakindex_arr, where=(leakindex_arr < 0), 
                         color='green', alpha=0.2, label='抑制区域 (<0)')
    
    ax_leak.set_xlabel('Epoch', fontsize=12)
    ax_leak.set_ylabel('LeakIndex (CDSC - CDDC)', fontsize=12)
    ax_leak.set_title('LeakIndex 演化 (带符号)', fontsize=14, fontweight='bold')
    ax_leak.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_leak.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    mean_leak = np.mean(leakindex_arr)
    std_leak = np.std(leakindex_arr)
    max_leak = np.max(leakindex_arr)
    min_leak = np.min(leakindex_arr)
    
    textstr_leak = '\n'.join([
        f'均值: {mean_leak:.4f}',
        f'标准差: {std_leak:.4f}',
        f'最大值: {max_leak:.4f}',
        f'最小值: {min_leak:.4f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_leak.text(
        0.02, 0.98, textstr_leak,
        transform=ax_leak.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
        fontproperties=FontProperties(family=_selected_font)
    )
    
    # === 右图: LeakIntensity (仅正值) ===
    ax_intensity = axes[1]
    ax_intensity.plot(epochs, leakintensity_arr, 'r-s', linewidth=2, markersize=6, label='LeakIntensity')
    ax_intensity.fill_between(epochs, 0, leakintensity_arr, color='red', alpha=0.2)
    ax_intensity.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_intensity.set_xlabel('Epoch', fontsize=12)
    ax_intensity.set_ylabel('LeakIntensity (max(0, LeakIndex))', fontsize=12)
    ax_intensity.set_title('LeakIntensity 演化 (仅正值)', fontsize=14, fontweight='bold')
    ax_intensity.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_intensity.grid(True, alpha=0.3)
    ax_intensity.set_ylim(bottom=0)  # 强制 y 轴从 0 开始
    
    # 添加统计信息文本框
    mean_intensity = np.mean(leakintensity_arr)
    std_intensity = np.std(leakintensity_arr)
    max_intensity = np.max(leakintensity_arr)
    
    textstr_intensity = '\n'.join([
        f'均值: {mean_intensity:.4f}',
        f'标准差: {std_intensity:.4f}',
        f'最大值: {max_intensity:.4f}',
        f'非零比例: {(leakintensity_arr > 1e-6).sum() / len(leakintensity_arr):.2%}'
    ])
    ax_intensity.text(
        0.02, 0.98, textstr_intensity,
        transform=ax_intensity.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props,
        fontproperties=FontProperties(family=_selected_font)
    )
    
    # 设置总标题
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存或显示
    _finalize_figure(fig, title, save_path)
    
    print(f"  📊 LeakIndex 曲线已保存: {save_path if save_path else _default_save_path(title)}")