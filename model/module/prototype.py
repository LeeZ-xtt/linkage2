import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeNetwork(nn.Module):
    """Few-shot 原型网络头"""
    def __init__(self, metric: str = 'euclidean'):
        super(PrototypeNetwork, self).__init__()
        self.metric = metric

    def compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor, n_way: int) -> torch.Tensor:
        """
        计算每个类别的原型（类别均值）
        
        Args:
            support_features: 支持集特征 [n_support, feature_dim]
            support_labels: 支持集标签 [n_support]
            n_way: 类别数
            
        Returns:
            prototypes: 类别原型 [n_way, feature_dim]
        """
        n_support = support_features.size(0)
        feature_dim = support_features.size(1)

        assert support_labels.min() >= 0, "标签必须从0开始"
        assert support_labels.max() < n_way, f"最大标签值{support_labels.max()}应小于类别数{n_way}"

        prototypes = torch.zeros(n_way, feature_dim, device=support_features.device)
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            if class_mask.sum() == 0:
                continue
            class_features = support_features[class_mask]
            prototypes[class_idx] = class_features.mean(dim=0)
        return prototypes

    def compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算查询特征与各类别原型之间的距离（或相似度）
        
        Args:
            query_features: 查询集特征 [n_query, feature_dim]
            prototypes: 类别原型 [n_way, feature_dim]
            
        Returns:
            distances: 查询样本到每个原型的距离/相似度 [n_query, n_way]
                      数值越大表示越相似（对于欧氏距离已取负，对于余弦即余弦相似度）
        """
        n_query = query_features.size(0)  # 查询样本数量
        n_way = prototypes.size(0)          # 类别（原型）数量
        
        if self.metric == 'euclidean':
            # 使用负欧氏距离平方作为相似度度量
            # torch.cdist 计算两两之间的 p=2 范数（欧氏距离），再取平方并取负
            # 负号保证数值越大表示越相似，与余弦相似度保持一致
            distances = -torch.cdist(query_features, prototypes, p=2) ** 2
        elif self.metric == 'cosine':
            # 使用余弦相似度作为度量
            # 先对查询特征和原型特征分别做 L2 归一化，使它们模长为 1
            qn = F.normalize(query_features, p=2, dim=1)
            pn = F.normalize(prototypes, p=2, dim=1)
            # 矩阵乘法得到余弦相似度，范围 [-1, 1]，数值越大越相似
            distances = torch.mm(qn, pn.t())
        else:
            # 若指定了未实现的度量方式，抛出异常
            raise ValueError(f"不支持的度量方式: {self.metric}")
        
        return distances

    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, query_features: torch.Tensor, n_way: int) -> torch.Tensor:
        """
        前向传播计算查询集的logits
        
        Args:
            support_features: 支持集特征 [n_support, feature_dim]
            support_labels: 支持集标签 [n_support]
            query_features: 查询集特征 [n_query, feature_dim]
            n_way: 类别数
            
        Returns:
            logits: 查询集logits [n_query, n_way]
            prototypes: 类别原型 [n_way, feature_dim]
        """
        prototypes = self.compute_prototypes(support_features, support_labels, n_way)
        logits = self.compute_distances(query_features, prototypes)
        return logits, prototypes