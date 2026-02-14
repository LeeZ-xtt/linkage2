"""
Semantic Anchor-driven Purification (SAP) 模块

通过交叉注意力机制，利用本征支计算的类别原型作为"语义锚点"，
显式识别并去除域特征中混入的类别语义成分，输出净化后的域特征。

设计目标:
    - 净化后的域特征应仅编码风格/纹理/域统计信息
    - 与语义特征正交，解决 CDSC-Leak 反弹问题

理论基础:
    - 交叉注意力机制 (Query-Key-Value)
    - 子空间投影净化
    - 残差净化 + 可学习门控

作者: Kiro AI Assistant
日期: 2026-02-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SemanticAnchorPurification(nn.Module):
    """
    语义锚点驱动净化模块
    
    通过交叉注意力计算域特征对各类别原型的相似度，
    重构并减除混入的语义成分，实现域特征净化。
    
    Args:
        dom_dim (int): 域特征维度 (默认 64)
        phy_dim (int): 本征语义特征维度 (默认 640)
        dropout (float): 投影层 dropout 率 (默认 0.1)
        normalize_query (bool): 是否对 Query 进行 L2 归一化 (默认 True)
        use_layernorm (bool): 是否对输出进行 LayerNorm (默认 True)
    """
    
    def __init__(
        self,
        dom_dim: int = 64,
        phy_dim: int = 640,
        dropout: float = 0.1,
        normalize_query: bool = True,
        use_layernorm: bool = True
    ):
        super().__init__()
        
        self.dom_dim = dom_dim
        self.phy_dim = phy_dim
        self.normalize_query = normalize_query
        self.use_layernorm = use_layernorm
        
        # Patch 3: 可学习的 logit scale (替代固定 temperature)
        # 用 log 参数化保证 scale > 0，初始值 25，上限 100
        logit_scale_init = 25.0
        self.logit_scale_log = nn.Parameter(torch.log(torch.tensor(logit_scale_init)))
        self.logit_scale_max = 100.0
        
        # Query 投影: 将域特征投影到语义空间
        self.W_q = nn.Sequential(
            nn.Linear(dom_dim, phy_dim, bias=True),
            nn.Dropout(dropout)
        )
        
        # 降维投影: 将语义成分投影回域特征空间
        self.W_proj = nn.Sequential(
            nn.Linear(phy_dim, dom_dim, bias=True),
            nn.Dropout(dropout)
        )
        
        # 正交损失投影 (无 bias，仅用于损失计算)
        self.W_orth = nn.Linear(dom_dim, phy_dim, bias=False)
        
        # 自适应净化强度门控 (可学习标量)
        # 初始化为 0.0，经 sigmoid 后为 0.5 (中等净化强度)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # 输出 LayerNorm (稳定净化后特征分布)
        if use_layernorm:
            self.output_ln = nn.LayerNorm(dom_dim)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化可学习参数
        
        策略:
            - W_q, W_proj: Xavier Uniform, gain=0.5 (避免初始注意力过于尖锐)
            - W_orth: Xavier Uniform, gain=1.0 (标准初始化)
            - alpha: 0.0 → sigmoid(0.0) = 0.5
            - LayerNorm: PyTorch 默认 (weight=1, bias=0)
        """
        # Query 投影和降维投影
        for module in [self.W_q, self.W_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # 正交损失投影
        nn.init.xavier_uniform_(self.W_orth.weight, gain=1.0)
    
    def compute_prototypes(
        self,
        f_phy: torch.Tensor,
        labels: torch.Tensor,
        n_way: int,
        k_shot: int
    ) -> torch.Tensor:
        """
        步骤 1: 计算语义锚点 (类别原型)
        
        基于 Support Set 的本征特征计算 N 个类别原型。
        
        Args:
            f_phy: 本征特征 [B, 640]
            labels: 类别标签 [B]
            n_way: 类别数 (N-way)
            k_shot: 每类支持样本数 (K-shot)
        
        Returns:
            prototypes: 归一化后的原型 [n_way, 640]
        """
        support_size = n_way * k_shot
        f_phy_support = f_phy[:support_size].detach()  # [N×K, 640] - 阻断梯度回流
        labels_support = labels[:support_size]  # [N×K]
        
        # 预分配原型张量
        prototypes = torch.zeros(n_way, self.phy_dim, device=f_phy.device)  # [n_way, 640]
        
        # 对每个类别计算原型
        for c in range(n_way):
            mask_c = (labels_support == c)  # 布尔掩码
            f_phy_c = f_phy_support[mask_c]  # [K, 640]
            prototypes[c] = f_phy_c.mean(dim=0)  # [640]
        
        # L2 归一化 (提升注意力计算稳定性)
        prototypes = F.normalize(prototypes, p=2, dim=1)  # [n_way, 640]
        
        return prototypes
    
    def compute_cross_attention(
        self,
        Q: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        步骤 2-3: 计算交叉注意力权重 (Patch 3: 可学习 scale)
        
        计算域特征 (Query) 与语义原型 (Key) 的相似度。
        
        Args:
            Q: Query 向量 [B, 640]
            prototypes: 语义原型 [n_way, 640]
        
        Returns:
            attn_weights: 注意力权重 [B, n_way]
            attn_logits: 注意力 logits [B, n_way] (用于诊断)
        """
        # 计算点积注意力
        attn_logits = Q @ prototypes.T  # [B, n_way]
        
        # Patch 3: 可学习 scale (替代除以 temperature)
        scale = torch.exp(self.logit_scale_log).clamp(max=self.logit_scale_max)  # 封顶
        attn_logits_scaled = attn_logits * scale  # [B, n_way]
        
        # 数值稳定性: row-wise 减最大值
        attn_logits_scaled = attn_logits_scaled - attn_logits_scaled.max(dim=-1, keepdim=True)[0]
        
        # Softmax 归一化
        attn_weights = F.softmax(attn_logits_scaled, dim=-1)  # [B, n_way]
        
        return attn_weights, attn_logits
    
    def reconstruct_semantic_component(
        self,
        attn_weights: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        步骤 4: 重构语义成分
        
        基于注意力权重，从语义原型中聚合出"应被去除的语义成分"。
        
        Args:
            attn_weights: 注意力权重 [B, n_way]
            prototypes: 语义原型 [n_way, 640]
        
        Returns:
            semantic_component_64: 降维后的语义成分 [B, 64]
        """
        # 在 640 维空间加权聚合
        semantic_component_640 = attn_weights @ prototypes  # [B, 640]
        
        # 投影回域特征空间
        semantic_component_64 = self.W_proj(semantic_component_640)  # [B, 64]
        
        return semantic_component_64
    
    def compute_purification_gate(self) -> torch.Tensor:
        """
        步骤 5: 计算自适应净化强度门控
        
        通过 Sigmoid 将可学习参数约束到 [0, 1] 范围。
        
        Returns:
            gate: 净化强度 (标量, 范围 [0, 1])
        """
        gate = torch.sigmoid(self.alpha)  # 标量
        return gate
    
    def apply_residual_purification(
        self,
        f_dom: torch.Tensor,
        semantic_component: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        步骤 6: 残差净化
        
        从原始域特征中减去经门控调制的语义成分。
        
        Args:
            f_dom: 原始域特征 [B, 64]
            semantic_component: 语义成分 [B, 64]
            gate: 净化强度 (标量)
        
        Returns:
            f_dom_pure: 净化后的域特征 [B, 64]
        """
        # 残差减除
        f_dom_pure = f_dom - gate * semantic_component  # [B, 64]
        
        # 可选: LayerNorm 稳定输出分布
        if self.use_layernorm:
            f_dom_pure = self.output_ln(f_dom_pure)
        
        return f_dom_pure
    
    def compute_orthogonal_loss(
        self,
        f_dom_pure: torch.Tensor,
        f_phy: torch.Tensor
    ) -> torch.Tensor:
        """
        步骤 7: 计算正交约束损失
        
        强制净化后的域特征与本征语义特征正交。
        
        Args:
            f_dom_pure: 净化后的域特征 [B, 64]
            f_phy: 本征语义特征 [B, 640]
        
        Returns:
            loss_orth: 正交损失 (标量)
        """
        # 投影到语义空间
        f_dom_pure_projected = self.W_orth(f_dom_pure)  # [B, 640]
        
        # 计算余弦相似度 (detach f_phy 防止影响本征支)
        cos_sim = F.cosine_similarity(
            f_dom_pure_projected,
            f_phy.detach(),
            dim=1
        )  # [B]
        
        # 正交损失: 余弦相似度的平方均值
        loss_orth = (cos_sim ** 2).mean()  # 标量
        
        return loss_orth
    
    def forward(
        self,
        f_dom: torch.Tensor,
        f_phy: torch.Tensor,
        labels: torch.Tensor,
        n_way: int,
        k_shot: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播: 完整的 SAP 净化流程
        
        Args:
            f_dom: 原始域特征 [B, 64]
            f_phy: 本征语义特征 [B, 640]
            labels: 类别标签 [B]
            n_way: 类别数 (N-way)
            k_shot: 每类支持样本数 (K-shot)
        
        Returns:
            f_dom_pure: 净化后的域特征 [B, 64]
            attn_weights: 交叉注意力权重 [B, n_way] (用于可视化)
            loss_orth: 正交约束损失 (标量)
            stats: 诊断统计信息 (dict)
        """
        B = f_dom.size(0)
        
        # 步骤 1: 计算语义锚点 (原型)
        prototypes = self.compute_prototypes(f_phy, labels, n_way, k_shot)  # [n_way, 640]
        
        # 步骤 2: 域特征投影 (Query 生成)
        Q = self.W_q(f_dom)  # [B, 640]
        if self.normalize_query:
            Q = F.normalize(Q, p=2, dim=1)  # L2 归一化
        
        # 步骤 3: 交叉注意力计算
        attn_weights, attn_logits = self.compute_cross_attention(Q, prototypes)  # [B, n_way]
        
        # 步骤 4: 语义成分重构
        semantic_component = self.reconstruct_semantic_component(attn_weights, prototypes)  # [B, 64]
        
        # 步骤 5: 自适应净化强度门控
        gate = self.compute_purification_gate()  # 标量
        
        # 步骤 6: 残差净化
        f_dom_pure = self.apply_residual_purification(f_dom, semantic_component, gate)  # [B, 64]
        
        # 步骤 7: 正交约束损失
        loss_orth = self.compute_orthogonal_loss(f_dom_pure, f_phy)  # 标量
        
        # Patch 3: 计算当前 logit_scale (用于监控)
        current_scale = torch.exp(self.logit_scale_log).clamp(max=self.logit_scale_max)
        scale_clipped = (current_scale >= self.logit_scale_max).float().mean().item()  # 触顶比例
        
        # 诊断统计信息
        stats = {
            'gate': gate.item(),  # 当前净化强度
            'attn_entropy': self._compute_entropy(attn_weights).mean().item(),  # 注意力熵
            'attn_max': attn_weights.max(dim=1)[0].mean().item(),  # 最大注意力均值
            'semantic_norm': semantic_component.norm(dim=1).mean().item(),  # 语义成分范数
            'purification_ratio': (gate * semantic_component.norm(dim=1) / (f_dom.norm(dim=1) + 1e-8)).mean().item(),  # 净化比例
            'w_orth_norm': self.W_orth.weight.norm().item(),  # W_orth 权重范数
            'logit_scale': current_scale.item(),  # Patch 3: 当前 scale
            'logit_scale_clipped': scale_clipped  # Patch 3: 触顶比例
        }
        
        return f_dom_pure, attn_weights, loss_orth, stats
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算概率分布的熵 (用于诊断注意力分布)
        
        Args:
            probs: 概率分布 [B, n_way]
        
        Returns:
            entropy: 熵值 [B]
        """
        # H(p) = -Σ p_i * log(p_i)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B]
        return entropy


# 模块配置信息
SAP_CONFIG = {
    'module_name': 'SemanticAnchorPurification',
    'description': '语义锚点驱动净化模块 - 通过交叉注意力去除域特征中的语义成分',
    'key_features': [
        '交叉注意力机制 (Query-Key-Value)',
        '子空间投影净化',
        '自适应净化强度门控',
        '正交约束损失',
        '残差净化 + LayerNorm'
    ],
    'default_hyperparameters': {
        'dom_dim': 64,
        'phy_dim': 640,
        'dropout': 0.1,
        'normalize_query': True,
        'use_layernorm': True,
        'temperature': 'sqrt(phy_dim)',
        'alpha_init': 0.0,
        'loss_weight': 0.1
    }
}
