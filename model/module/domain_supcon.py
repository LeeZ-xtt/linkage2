"""
域监督对比学习模块 (Domain Supervised Contrastive Learning)

核心目标: 让 f_dom 成为「域统计探针」
- 同域样本（无论类别）在特征空间中聚拢
- 异域样本分离
- 主动压制类别判别能力

核心设计:
- DomainProjectionHead: 128D → 64D 对比空间 (GELU + L2归一化)
- InverseUTemperatureScheduler: 倒 U 形温度调度 (0.2 → 0.3 → 0.15)
- DomainSupConLoss: 同域异类加权的域对比损失

参考: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DomainProjectionHead(nn.Module):
    """
    域特征投影头 (Domain Projection Head)
    
    将域编码器的特征投影到对比学习空间（默认: 64D → 64D）。
    采用轻量瓶颈结构: in_dim → hidden_dim → out_dim，配合 GELU 和较高 Dropout。
    
    设计原则:
    - 输入维度较小（128），投影头应更轻量
    - 使用 GELU 替代 LeakyReLU，符合现代 Transformer 偏好
    - 较高 Dropout 率（0.15），防止域对比目标塌缩
    """
    
    def __init__(
        self, 
        in_dim: int = 64, 
        hidden_dim: int = 32, 
        out_dim: int = 32,
        dropout: float = 0.15
    ) -> None:
        """
        Args:
            in_dim: 输入特征维度 (域编码器输出)
            hidden_dim: 隐藏层维度
            out_dim: 输出投影维度 (对比空间)
            dropout: Dropout 率
        """
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )
        
        # 权重初始化 (截断正态分布，std=0.02)
        self._init_weights()
        
    def _init_weights(self) -> None:
        """截断正态分布初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 域特征 [B, 128]
            
        Returns:
            z: L2 归一化的投影特征 [B, 64]
        """
        z = self.projector(x)  # [B, 64]
        z = F.normalize(z, p=2, dim=1)  # L2 归一化到单位超球面
        return z


class InverseUTemperatureScheduler:
    """
    倒 U 形温度调度器 (Inverse-U Temperature Scheduler)
    
    域对比学习采用「快启动 → 放松 → 精调」模式:
    - 早期低温 (τ=0.2): 快速建立域聚类结构
    - 中期高温 (τ=0.3): 避免域内塌缩，保留多样性
    - 后期低温 (τ=0.15): 精细化域边界
    
    这种调度策略与 HuggingFace Trainer 的学习率调度类似
    (warmup → constant → decay)
    """
    
    def __init__(
        self,
        early_temp: float = 0.2,
        mid_temp: float = 0.3,
        final_temp: float = 0.15,
        early_epochs: int = 10,
        mid_epochs: int = 60,
        total_epochs: int = 100
    ) -> None:
        """
        Args:
            early_temp: 早期温度 (快速聚类)
            mid_temp: 中期温度 (保留多样性)
            final_temp: 后期温度 (精细化边界)
            early_epochs: 早期阶段结束 epoch
            mid_epochs: 中期阶段结束 epoch
            total_epochs: 总训练 epoch 数
        """
        self.early_temp = early_temp
        self.mid_temp = mid_temp
        self.final_temp = final_temp
        self.early_epochs = early_epochs
        self.mid_epochs = mid_epochs
        self.total_epochs = total_epochs
        
    def get_temperature(self, epoch: int) -> float:
        """
        根据当前 epoch 获取温度值
        
        Args:
            epoch: 当前 epoch (0-indexed)
            
        Returns:
            当前温度值
        """
        if epoch < self.early_epochs:
            # 早期: 保持低温，快速建立域聚类
            return self.early_temp
        elif epoch < self.mid_epochs:
            # 中期: 升温到 mid_temp，允许域内多样性
            progress = (epoch - self.early_epochs) / max(self.mid_epochs - self.early_epochs, 1)
            return self.early_temp + (self.mid_temp - self.early_temp) * min(progress, 1.0)
        else:
            # 后期: 降温到 final_temp，精细化域边界
            progress = (epoch - self.mid_epochs) / max(self.total_epochs - self.mid_epochs, 1)
            return self.mid_temp - (self.mid_temp - self.final_temp) * min(progress, 1.0)


class DomainSupConLoss(nn.Module):
    """
    域监督对比损失 (Domain Supervised Contrastive Loss)
    
    数学形式:
        L = -1/N ∑ᵢ [ 1/|P(i)| ∑ₚ∈P(i) wₚ · log(exp(zᵢ·zₚ/τ) / ∑ₐ≠ᵢ exp(zᵢ·zₐ/τ)) ]
    
    其中:
    - P(i): 锚点 i 的同域样本集
    - wₚ: 正样本权重 (同域异类=1.5, 同域同类=0.3)
    - τ: 倒 U 形温度调度 (0.2 → 0.3 → 0.15)
    
    目标: 让 f_dom 只编码域统计信息，压制类别可分性
    """
    
    def __init__(
        self,
        in_dim: int = 64,
        proj_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        early_temp: float = 0.2,
        mid_temp: float = 0.3,
        final_temp: float = 0.15,
        early_epochs: int = 10,
        mid_epochs: int = 60,
        total_epochs: int = 100,
        cross_class_weight: float = 1.5,
        same_class_weight: float = 0.3,
        cdsc_neg_weight: float = 1.0
    ) -> None:
        """
        Args:
            in_dim: 域特征维度
            proj_dim: 投影空间维度
            hidden_dim: 投影头隐藏层维度
            dropout: 投影头 Dropout 率
            early_temp: 早期温度
            mid_temp: 中期温度
            final_temp: 后期温度
            early_epochs: 早期阶段结束 epoch
            mid_epochs: 中期阶段结束 epoch
            total_epochs: 总训练 epoch 数
            cross_class_weight: 同域异类正样本权重 (>1.0)
            same_class_weight: 同域同类正样本权重 (<1.0)
            cdsc_neg_weight: 跨域同类负样本权重 (>1.0, 默认1.0表示无加权)
        """
        super().__init__()
        
        self.projection_head = DomainProjectionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim,
            dropout=dropout
        )
        
        # 温度调度器 (倒 U 形)
        self.temp_scheduler = InverseUTemperatureScheduler(
            early_temp=early_temp,
            mid_temp=mid_temp,
            final_temp=final_temp,
            early_epochs=early_epochs,
            mid_epochs=mid_epochs,
            total_epochs=total_epochs
        )
        
        # 正样本权重配置
        self.cross_class_weight = cross_class_weight  # 同域异类权重
        self.same_class_weight = same_class_weight    # 同域同类权重
        
        # 困难负样本权重配置
        self.cdsc_neg_weight = cdsc_neg_weight  # 跨域同类负样本权重
        
        # 参数验证
        assert cdsc_neg_weight >= 1.0, \
            f"cdsc_neg_weight 必须 >= 1.0, 当前值: {cdsc_neg_weight}"
        
    def forward(
        self,
        domain_features: torch.Tensor,
        domain_labels: torch.Tensor,
        class_labels: torch.Tensor,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算域 SupCon 损失
        
        Args:
            domain_features: 域特征 [N, 128]
            domain_labels: 域标签 [N]
            class_labels: 类别标签 [N] (用于构建加权正样本)
            epoch: 当前 epoch (用于温度调度)
            
        Returns:
            loss: 域 SupCon 损失
            stats: 统计信息
        """
        device = domain_features.device
        batch_size = domain_features.shape[0]
        
        # 投影到对比空间 (L2 归一化后的特征位于单位超球面)
        z = self.projection_head(domain_features)  # [N, proj_dim], 已L2归一化
        
        # 获取当前温度 (倒 U 形调度)
        temperature = self.temp_scheduler.get_temperature(epoch)
        
        # 构建同域掩码: same_domain_mask[i,j]=1 表示样本 i 和 j 来自同一域
        domain_col = domain_labels.view(-1, 1)  # [N, 1]
        same_domain_mask = torch.eq(domain_col, domain_labels.view(1, -1)).float()  # [N, N], 对称矩阵
        
        # 移除对角线 (自己不能是自己的正样本)
        self_mask = torch.eye(batch_size, device=device)
        same_domain_mask = same_domain_mask * (1 - self_mask)  # [N, N]
        
        # 构建同类掩码: same_class_mask[i,j]=1 表示样本 i 和 j 属于同一类别
        class_col = class_labels.view(-1, 1)  # [N, 1]
        same_class_mask = torch.eq(class_col, class_labels.view(1, -1)).float()  # [N, N], 对称矩阵
        
        # 计算损失
        loss, stats = self._compute_domain_supcon_loss(
            z, same_domain_mask, same_class_mask, self_mask, temperature
        )
        
        return loss, stats
    
    def _compute_domain_supcon_loss(
        self,
        features: torch.Tensor,
        same_domain_mask: torch.Tensor,
        same_class_mask: torch.Tensor,
        self_mask: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算带加权的域 SupCon 损失
        
        正样本权重策略:
        - 同域 + 异类: w = cross_class_weight (1.5)  → 强拉近
        - 同域 + 同类: w = same_class_weight (0.3)  → 弱拉近，避免类别聚类
        - 异域: w = 0 (负样本)
        
        Args:
            features: 已 L2 归一化的特征 [N, D]
            same_domain_mask: 同域掩码 [N, N] (已移除对角线)
            same_class_mask: 同类掩码 [N, N]
            self_mask: 自身掩码 [N, N] (对角线为1)
            temperature: 温度参数 τ
            
        Returns:
            loss: 加权域 SupCon 损失
            stats: 统计信息
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 计算相似度矩阵 (features 已 L2 归一化，内积即余弦相似度)
        raw_sim_matrix = torch.mm(features, features.T)  # [N, N], 值域 [-1, 1]
        
        # 温度缩放的相似度
        sim_matrix = raw_sim_matrix / temperature  # [N, N]
        
        # 数值稳定性: 减去每行最大值 (LogSumExp trick)
        sim_for_max = sim_matrix.masked_fill(self_mask.bool(), float('-inf'))
        sim_max, _ = sim_for_max.max(dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max.detach()  # [N, N]
        
        # 计算 log(sum(exp)) 分母 (排除自己)
        exp_sim = torch.exp(sim_matrix_stable) * (1 - self_mask)  # [N, N]
        
        # 困难负样本加权: 跨域同类 (CDSC)
        # 目标: 让域编码器不依赖类别信息进行域分类
        if self.cdsc_neg_weight != 1.0:
            # 负样本掩码 = 非同域 & 非自身
            # 注意: 在域对比学习中，异域样本是负样本
            neg_mask = (1.0 - same_domain_mask) * (1.0 - self_mask)  # [N, N]
            
            # CDSC负样本 = 负样本 & 同类
            # 这些样本来自不同域但属于同一类别，是"困难负样本"
            cdsc_neg_mask = neg_mask * same_class_mask  # [N, N]
            
            # 内存优化: 直接在 exp_sim 上应用权重
            # exp_new = exp_old + exp_old * mask * (w - 1)
            weight_delta = self.cdsc_neg_weight - 1.0  # scalar
            weighted_exp_sim = exp_sim + exp_sim * cdsc_neg_mask * weight_delta  # [N, N]
            
            log_sum_exp = torch.log(weighted_exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [N, 1]
        else:
            # 标准计算（无加权）
            log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [N, 1]
        
        # 构建正样本对权重: 同域异类=1.5, 同域同类=0.3
        # 异类掩码: diff_class[i,j]=1 表示 i 和 j 类别不同
        diff_class_mask = 1 - same_class_mask  # [N, N]
        
        # 正样本权重矩阵 (仅同域样本为正样本)
        # 同域异类: cross_class_weight, 同域同类: same_class_weight
        pos_weights = same_domain_mask * (
            diff_class_mask * self.cross_class_weight +
            same_class_mask * self.same_class_weight
        )  # [N, N]
        
        # 正样本数量 (同域样本数，不是权重和)
        pos_count = same_domain_mask.sum(dim=1)  # [N]
        
        # 计算 log 概率
        log_prob = sim_matrix_stable - log_sum_exp  # [N, N]
        
        # 加权的 log 概率 (同域异类贡献更大梯度)
        weighted_log_prob = log_prob * pos_weights  # [N, N]
        
        # 对每个锚点计算损失: -∑(w_p * log_prob_p) / |P(i)|
        # 注意: 除以正样本数量（而非权重和），保持损失尺度一致
        valid_anchors = pos_count > 0
        loss_per_anchor = torch.zeros(batch_size, device=device)
        loss_per_anchor[valid_anchors] = (
            -weighted_log_prob[valid_anchors].sum(dim=1) / pos_count[valid_anchors]
        )
        
        # 平均损失
        num_valid = valid_anchors.sum()
        if num_valid > 0:
            loss = loss_per_anchor.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 统计信息 (使用原始相似度矩阵)
        with torch.no_grad():
            # 同域异类相似度 (关键指标)
            cross_class_pos_mask = same_domain_mask * diff_class_mask
            cross_class_sim_sum = (raw_sim_matrix * cross_class_pos_mask).sum()
            cross_class_count = cross_class_pos_mask.sum()
            avg_cross_class_sim = cross_class_sim_sum / (cross_class_count + 1e-8)
            
            # 同域同类相似度
            same_class_pos_mask = same_domain_mask * same_class_mask
            same_class_sim_sum = (raw_sim_matrix * same_class_pos_mask).sum()
            same_class_count = same_class_pos_mask.sum()
            avg_same_class_sim = same_class_sim_sum / (same_class_count + 1e-8)
            
            # 异域相似度 (负样本) - 细分为 Cross-Domain/Same-Class 和 Cross-Domain/Cross-Class
            diff_domain_mask = (1 - same_domain_mask) * (1 - self_mask)
            
            # 1. Cross-Domain / Same-Class (CDSC) - 关键监控指标: 泄露检测
            cdsc_mask = diff_domain_mask * same_class_mask
            cdsc_sim_sum = (raw_sim_matrix * cdsc_mask).sum()
            cdsc_count = cdsc_mask.sum()
            avg_cdsc_sim = cdsc_sim_sum / (cdsc_count + 1e-8)
            
            # 2. Cross-Domain / Cross-Class (CDDC) - 纯负样本
            cddc_mask = diff_domain_mask * diff_class_mask
            cddc_sim_sum = (raw_sim_matrix * cddc_mask).sum()
            cddc_count = cddc_mask.sum()
            avg_cddc_sim = cddc_sim_sum / (cddc_count + 1e-8)
            
            # 3. CDSC负样本相似度 (新增监控指标)
            # 用于监控困难负样本加权的效果
            cdsc_neg_mask_stats = diff_domain_mask * same_class_mask  # [N, N]
            cdsc_neg_sim_sum = (raw_sim_matrix * cdsc_neg_mask_stats).sum()
            cdsc_neg_count = cdsc_neg_mask_stats.sum()
            avg_cdsc_neg_sim = cdsc_neg_sim_sum / (cdsc_neg_count + 1e-8)
        
        stats = {
            'avg_cross_class_sim': avg_cross_class_sim.item(),  # SDDC (同域异类) —— 应该低（异类距离大）
            'avg_same_class_sim': avg_same_class_sim.item(),    # SDSC (同域同类) —— 应该高（同类距离小）
            'avg_cdsc_sim': avg_cdsc_sim.item(),                # CDSC (跨域同类) —— 应该高（同类跨域也要靠近）
            'avg_cddc_sim': avg_cddc_sim.item(),                # CDDC (跨域异类) —— 应该低（异类跨域距离大）
            'avg_cdsc_neg_sim': avg_cdsc_neg_sim.item(),       # CDSC负样本相似度 (新增)
            'cdsc_neg_pairs': int(cdsc_neg_count.item()),      # CDSC负样本对数量 (新增)
            'num_valid_anchors': int(num_valid.item()),
            'cross_class_pairs': int(cross_class_count.item()),
            'same_class_pairs': int(same_class_count.item()),
            'temperature': temperature,
            'cdsc_neg_weight': self.cdsc_neg_weight  # 新增: 权重值
        }
        
        return loss, stats


def create_domain_supcon_module(
    in_dim: int = 64,
    proj_dim: int = 32,
    hidden_dim: int = 32,
    dropout: float = 0.15,
    total_epochs: int = 100
) -> DomainSupConLoss:
    """
    工厂函数: 创建域 SupCon 模块
    
    Args:
        in_dim: 域特征维度
        proj_dim: 投影空间维度
        hidden_dim: 投影头隐藏层维度
        dropout: 投影头 Dropout 率
        total_epochs: 总训练 epoch 数
        
    Returns:
        DomainSupConLoss 实例
    """
    return DomainSupConLoss(
        in_dim=in_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        total_epochs=total_epochs
    )
