"""
本征支监督对比学习模块 (Intrinsic Supervised Contrastive Learning)

核心目标: 让 f_phy 学习「类别判别」特征
- 同类别样本在特征空间中聚拢
- 不同类别样本分离
- 跨域正样本对给予更高权重，强化域不变性
- 同域正样本对给予较低权重，避免域捷径 (domain shortcut)

核心组件:
- IntrinsicProjectionHead: 640D → 128D 对比空间 (GELU + L2归一化)
- PhasedTemperatureScheduler: 分段温度调度 
- IntrinsicSupConLoss: 域感知加权的 SupCon 损失

参考: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IntrinsicProjectionHead(nn.Module):
    """
    本征特征投影头 (Intrinsic Projection Head)
    
    将本征编码器的 640D 特征投影到 128D 的对比学习空间。
    采用瓶颈结构: 640 → 256 → 128，配合 GELU 和 Dropout。
    """
    
    def __init__(
        self, 
        in_dim: int = 640, 
        hidden_dim: int = 256, 
        out_dim: int = 128,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )
        self._init_weights()
        
    def _init_weights(self) -> None:
        """截断正态分布初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: [B, 640] → L2归一化的 [B, 128]"""
        z = self.projector(x)  # [B, 128]
        z = F.normalize(z, p=2, dim=1)
        return z


class PhasedTemperatureScheduler:
    """
    分段温度调度器 (Phased Temperature Scheduler)
    
    与域支倒 U 形调度协同:
    - 本征支: 0.5 → 0.3 → 0.2 (早期高温等待域支 → 中期对称平台 → 后期强收紧)
    """
    
    def __init__(
        self,
        phase1_temp: float = 0.5,
        phase2_temp: float = 0.3,
        phase3_temp: float = 0.2,
        phase1_end: int = 15,
        phase2_start: int = 30,
        phase2_end: int = 70,
        total_epochs: int = 100
    ) -> None:
        self.phase1_temp = phase1_temp
        self.phase2_temp = phase2_temp
        self.phase3_temp = phase3_temp
        self.phase1_end = phase1_end
        self.phase2_start = phase2_start
        self.phase2_end = phase2_end
        self.total_epochs = total_epochs
        
    def get_temperature(self, epoch: int) -> float:
        """根据当前 epoch 获取温度值"""
        if epoch <= self.phase1_end:
            return self.phase1_temp
        elif epoch < self.phase2_start:
            progress = (epoch - self.phase1_end) / max(self.phase2_start - self.phase1_end, 1)
            return self.phase1_temp - (self.phase1_temp - self.phase2_temp) * progress
        elif epoch <= self.phase2_end:
            return self.phase2_temp
        else:
            progress = (epoch - self.phase2_end) / max(self.total_epochs - self.phase2_end, 1)
            return self.phase2_temp - (self.phase2_temp - self.phase3_temp) * min(progress, 1.0)


class IntrinsicSupConLoss(nn.Module):
    """
    本征支监督对比损失 (Intrinsic Supervised Contrastive Loss)
    
    数学形式:
        L = -1/N ∑ᵢ [ 1/|P(i)| ∑ₙ∈P(i) wₙ · log(exp(zᵢ·zₙ/τ) / ∑ₐ≠ᵢ exp(zᵢ·zₐ/τ)) ]
    
    正样本加权策略:
    - 跨域同类: w=1.5 → 强化域不变性
    - 同域同类: w=0.8 → 避免域捷径
    """
    
    def __init__(
        self,
        in_dim: int = 640,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        phase1_temp: float = 0.5,
        phase2_temp: float = 0.3,
        phase3_temp: float = 0.2,
        phase1_end: int = 15,
        phase2_start: int = 30,
        phase2_end: int = 70,
        total_epochs: int = 100,
        cross_domain_weight: float = 1.5,
        same_domain_weight: float = 0.8,
        cross_domain_neg_weight: float = 1.2
    ) -> None:
        super().__init__()
        
        self.projection_head = IntrinsicProjectionHead(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=proj_dim, dropout=dropout
        )
        
        self.temp_scheduler = PhasedTemperatureScheduler(
            phase1_temp=phase1_temp, phase2_temp=phase2_temp, phase3_temp=phase3_temp,
            phase1_end=phase1_end, phase2_start=phase2_start, phase2_end=phase2_end,
            total_epochs=total_epochs
        )
        
        self.cross_domain_weight = cross_domain_weight
        self.same_domain_weight = same_domain_weight
        self.cross_domain_neg_weight = cross_domain_neg_weight
        
    def forward(
        self,
        intrinsic_features: torch.Tensor,
        class_labels: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算本征 SupCon 损失
        
        Args:
            intrinsic_features: 本征特征 [N, 640]
            class_labels: 类别标签 [N]
            domain_labels: 域标签 [N] (可选，用于域感知加权)
            epoch: 当前 epoch (用于温度调度)
            
        Returns:
            loss: SupCon 损失
            stats: 统计信息
        """
        device = intrinsic_features.device
        batch_size = intrinsic_features.shape[0]
        
        # 投影到对比空间 (L2 归一化后位于单位超球面)
        z = self.projection_head(intrinsic_features)  # [N, proj_dim]
        
        # 获取当前温度
        temperature = self.temp_scheduler.get_temperature(epoch)
        
        # 构建同类掩码
        class_col = class_labels.view(-1, 1)  # [N, 1]
        same_class_mask = torch.eq(class_col, class_labels.view(1, -1)).float()  # [N, N]
        
        # 移除对角线
        self_mask = torch.eye(batch_size, device=device)
        same_class_mask = same_class_mask * (1 - self_mask)
        
        # 构建跨域掩码 (如果提供域标签)
        cross_domain_mask = None
        if domain_labels is not None:
            domain_col = domain_labels.view(-1, 1)  # [N, 1]
            cross_domain_mask = (domain_col != domain_labels.view(1, -1)).float()  # [N, N]
        
        # 计算损失
        loss, stats = self._compute_intrinsic_supcon_loss(
            z, same_class_mask, cross_domain_mask, self_mask, temperature
        )
        
        return loss, stats
    
    def _compute_intrinsic_supcon_loss(
        self,
        features: torch.Tensor,
        same_class_mask: torch.Tensor,
        cross_domain_mask: Optional[torch.Tensor],
        self_mask: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算带域感知权重的 SupCon 损失
        
        正样本权重策略:
        - 跨域同类: cross_domain_weight (1.5) → 强化域不变性
        - 同域同类: same_domain_weight (0.8) → 避免域捷径
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 计算相似度矩阵 (features 已 L2 归一化)
        raw_sim_matrix = torch.mm(features, features.T)  # [N, N]
        
        # 温度缩放
        sim_matrix = raw_sim_matrix / temperature
        
        # 数值稳定性: LogSumExp trick
        sim_for_max = sim_matrix.masked_fill(self_mask.bool(), float('-inf'))
        sim_max, _ = sim_for_max.max(dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max.detach()
        
        # 构建正样本权重
        if cross_domain_mask is not None:
            same_domain_mask = 1.0 - cross_domain_mask
            pos_weights = same_class_mask * (
                self.cross_domain_weight * cross_domain_mask +
                self.same_domain_weight * same_domain_mask
            )
        else:
            pos_weights = same_class_mask.clone()
        
        # 计算 log(sum(exp)) 分母
        exp_sim = torch.exp(sim_matrix_stable) * (1 - self_mask)
        
        # 引入困难负样本权重 (跨域异类)
        # Best Practice: 通过在分母中放大特定负样本的 exp 值，显式增加其梯度贡献 (Gradient Scaling)
        if cross_domain_mask is not None and self.cross_domain_neg_weight != 1.0:
            # 负样本掩码 = 非同类 & 非自身
            neg_mask = (1.0 - same_class_mask) * (1.0 - self_mask)
            # 跨域负样本 = 负样本 & 跨域
            cross_domain_neg_mask = neg_mask * cross_domain_mask
            
            # 内存优化: 直接在 exp_sim 上应用权重，避免创建额外的权重矩阵
            # exp_new = exp_old + exp_old * mask * (w - 1)
            weight_delta = self.cross_domain_neg_weight - 1.0
            weighted_exp_sim = exp_sim + exp_sim * cross_domain_neg_mask * weight_delta
            
            log_sum_exp = torch.log(weighted_exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # 标准计算
            log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 正样本数量
        pos_count = same_class_mask.sum(dim=1)  # [N]
        
        # 计算加权 log 概率
        log_prob = sim_matrix_stable - log_sum_exp
        weighted_log_prob = log_prob * pos_weights
        
        # 计算损失
        valid_anchors = pos_count > 0
        loss_per_anchor = torch.zeros(batch_size, device=device)
        loss_per_anchor[valid_anchors] = (
            -weighted_log_prob[valid_anchors].sum(dim=1) / pos_count[valid_anchors]
        )
        
        num_valid = valid_anchors.sum()
        if num_valid > 0:
            loss = loss_per_anchor.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 统计信息
        with torch.no_grad():
            # 跨域同类相似度 (关键指标)
            if cross_domain_mask is not None:
                cross_class_pos_mask = same_class_mask * cross_domain_mask
                cross_class_sim_sum = (raw_sim_matrix * cross_class_pos_mask).sum()
                cross_class_count = cross_class_pos_mask.sum()
                avg_cross_domain_sim = cross_class_sim_sum / (cross_class_count + 1e-8)
                
                same_class_pos_mask = same_class_mask * (1.0 - cross_domain_mask)
                same_class_sim_sum = (raw_sim_matrix * same_class_pos_mask).sum()
                same_class_count = same_class_pos_mask.sum()
                avg_same_domain_sim = same_class_sim_sum / (same_class_count + 1e-8)
            else:
                pos_sim_sum = (raw_sim_matrix * same_class_mask).sum()
                pos_count_total = same_class_mask.sum()
                avg_cross_domain_sim = pos_sim_sum / (pos_count_total + 1e-8)
                avg_same_domain_sim = avg_cross_domain_sim
                cross_class_count = pos_count_total
                same_class_count = torch.tensor(0.0)
            
            # 负样本相似度
            neg_mask = (1 - same_class_mask) * (1 - self_mask)
            neg_sim_sum = (raw_sim_matrix * neg_mask).sum()
            neg_count = neg_mask.sum()
            avg_neg_sim = neg_sim_sum / (neg_count + 1e-8)
        
        stats = {
            'avg_cross_domain_sim': avg_cross_domain_sim.item(),  # 跨域同类相似度 (应该高)
            'avg_same_domain_sim': avg_same_domain_sim.item(),    # 同域同类相似度
            'avg_neg_sim': avg_neg_sim.item(),                    # 负样本相似度 (应该低)
            'num_valid_anchors': int(num_valid.item()),
            'cross_domain_pairs': int(cross_class_count.item()),
            'same_domain_pairs': int(same_class_count.item()),
            'temperature': temperature
        }
        
        return loss, stats


def create_intrinsic_supcon_module(
    in_dim: int = 640,
    proj_dim: int = 128,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    total_epochs: int = 100
) -> IntrinsicSupConLoss:
    """
    工厂函数: 创建本征支 SupCon 模块
    
    Args:
        in_dim: 本征特征维度
        proj_dim: 投影空间维度
        hidden_dim: 投影头隐藏层维度
        dropout: 投影头 Dropout 率
        total_epochs: 总训练 epoch 数
        
    Returns:
        IntrinsicSupConLoss 实例
    """
    return IntrinsicSupConLoss(
        in_dim=in_dim, proj_dim=proj_dim, hidden_dim=hidden_dim,
        dropout=dropout, total_epochs=total_epochs
    )


# ============ 单元测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("测试 IntrinsicSupConLoss 模块")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 模拟 5-way 5-shot 15-query 场景
    n_way, k_shot, q_per_class = 5, 5, 15
    n_support = n_way * k_shot  # 25
    n_query = n_way * q_per_class  # 75
    total = n_support + n_query  # 100
    
    # 模拟本征特征
    intrinsic_features = torch.randn(total, 640).to(device)
    
    # 生成类别标签
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query_labels = torch.arange(n_way).repeat_interleave(q_per_class)
    class_labels = torch.cat([support_labels, query_labels]).to(device)
    
    # 生成域标签: 支持集来自域 0/1/2, 查询集来自域 3
    import numpy as np
    np.random.seed(42)
    support_domains = torch.tensor(np.random.choice([0, 1, 2], size=n_support))
    query_domains = torch.full((n_query,), 3, dtype=torch.long)
    domain_labels = torch.cat([support_domains, query_domains]).to(device)
    
    # 创建模块
    supcon_module = create_intrinsic_supcon_module().to(device)
    
    # 测试温度调度 (分段式)
    print("温度调度测试 (分段式):")
    for ep in [0, 15, 25, 50, 70, 85, 100]:
        temp = supcon_module.temp_scheduler.get_temperature(ep)
        print(f"  Epoch {ep:3d}: τ = {temp:.3f}")
    
    # 测试损失计算 (Epoch=50)
    print("\n损失计算测试 (Epoch=50):")
    loss, stats = supcon_module(
        intrinsic_features, class_labels, domain_labels, epoch=50
    )
    print(f"  Loss: {loss.item():.4f}")
    print(f"  温度: {stats['temperature']:.3f}")
    print(f"  跨域同类相似度: {stats['avg_cross_domain_sim']:.4f}")
    print(f"  同域同类相似度: {stats['avg_same_domain_sim']:.4f}")
    print(f"  负样本相似度: {stats['avg_neg_sim']:.4f}")
    print(f"  跨域正样本对数: {stats['cross_domain_pairs']}")
    print(f"  同域正样本对数: {stats['same_domain_pairs']}")
    
    # 测试梯度
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in supcon_module.parameters() if p.grad is not None) ** 0.5
    print(f"\n梯度范数: {grad_norm:.4f}")
    print("✅ 梯度测试通过!")
