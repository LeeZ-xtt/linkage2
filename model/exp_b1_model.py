"""
Experiment B.1 Model - Dual Stream + Domain Classifier + HSIC

该模型旨在验证特征解耦的有效性。
包含:
1. 双流主干 (DualStreamResNet12)
2. 原型分类头 (PrototypeNetwork)
3. 域分类器 (Domain Classifier)
4. HSIC 损失计算

使用方法:
    from model.exp_b1_model import ExpB1Model
    model = ExpB1Model()
    logits, prototypes, domain_logits, hsic_loss = model(support_images, support_labels, query_images, n_way)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.intrinsic_encoder_v2 import IntrinsicEncoder
from model.backbone.style_stat_domain_encoder import StyleStatDomainEncoder, StyleStatDomainEncoderConfig
from model.module.prototype import PrototypeNetwork
from model.module.intrinsic_supcon import IntrinsicSupConLoss
from model.module.domain_supcon import DomainSupConLoss
from model.module.semantic_anchor_purification import SemanticAnchorPurification


class DualStreamResNet12(nn.Module):
    """
    双流主干网络 (Dual-Stream Backbone)
    
    将本征编码器与域编码器组合，对外保持与旧代码完全一致的接口。
    """

    def __init__(
        self,
        avg_pool: bool = True,
        drop_rate: float = 0.1,
        # 域支：显式风格统计编码（方案A）
        domain_cfg: StyleStatDomainEncoderConfig | None = None,
    ) -> None:
        super(DualStreamResNet12, self).__init__()
        # 本征分支: InstanceNorm + 可选空洞卷积
        self.intrinsic_encoder = IntrinsicEncoder(drop_rate=drop_rate, dilated=True)
        # 域分支: 固定滤波器组 + 显式统计（禁用 BN，抑制语义捷径）
        self.domain_encoder = StyleStatDomainEncoder(cfg=domain_cfg)

        # 保留旧属性，确保外部调用无需改动
        self.feature_dim = self.intrinsic_encoder.feature_dim  # 640
        self.domain_dim = self.domain_encoder.feature_dim      # domain_dim (默认 64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            f_phy: 本征特征 [B, 640]
            f_dom: 域特征   [B, domain_dim]
        """
        # 提取本征特征（空间特征图）
        f_phy_map = self.intrinsic_encoder(x)
        # 全局平均池化并展平 -> [B, 640]
        f_phy = nn.functional.adaptive_avg_pool2d(f_phy_map, (1, 1)).flatten(1)

        # 提取域特征
        f_dom = self.domain_encoder(x)

        return f_phy, f_dom

class ExpB1Model(nn.Module):
    """
    实验 B.1 模型: 双流网络 + 域分类器
    """
    def __init__(self, n_domains=4, metric='euclidean', 
                 domain_dropout=0.5, intrinsic_encoder_drop_rate: float = 0.2,
                 use_intrinsic_supcon=True, intrinsic_supcon_weight=0.2, intrinsic_supcon_proj_dim=128,
                 intrinsic_supcon_dropout=0.1,
                 # 本征支分段温度调度 (与域支协同)
                 intrinsic_supcon_phase1_temp=0.5, intrinsic_supcon_phase2_temp=0.3, intrinsic_supcon_phase3_temp=0.2,
                 intrinsic_supcon_phase1_end=15, intrinsic_supcon_phase2_start=30, intrinsic_supcon_phase2_end=70,
                 # 本征支正/负样本权重
                 intrinsic_supcon_cross_domain_weight=1.5, intrinsic_supcon_same_domain_weight=0.8,
                 intrinsic_supcon_cross_domain_neg_weight=1.2,
                 total_epochs=100,
                 # 域对比学习参数
                 use_domain_supcon=True, domain_supcon_weight=0.08,
                 domain_supcon_proj_dim=32, domain_supcon_dropout=0.15,
                 domain_supcon_early_temp=0.2, domain_supcon_mid_temp=0.3,
                 domain_supcon_final_temp=0.15, domain_supcon_early_epochs=10,
                 domain_supcon_mid_epochs=60,
                 domain_supcon_cross_class_weight=1.5, domain_supcon_same_class_weight=0.3,
                 domain_supcon_cdsc_neg_weight=1.0,
                 # ==========================
                 # SAP 模块配置
                 # ==========================
                 use_sap=True, sap_dropout=0.1, sap_orth_weight=0.1, k_shot=5,
                 # ==========================
                 # 域支风格统计编码器配置（方案A）
                 # ==========================
                 domain_dim: int = 64,
                 domain_filterbank_kernel: int = 7,
                 domain_filterbank_use_gabor: bool = True,
                 domain_filterbank_use_dog: bool = True,
                 domain_filterbank_trainable_gain: bool = True,
                 domain_filterbank_trainable_delta_kernel: bool = False,
                 domain_token_dropout: float = 0.4,
                 domain_token_scale_init: float = 1.0,
                 domain_use_gram_stats: bool = True,
                 domain_gram_rank: int = 16,
                 domain_gram_out_dim: int = 64,
                 domain_norm: str = "gn"):
        """
        初始化ExpB1Model - 双流架构 + 域分类器 + 双向对比学习

        Args:
            n_domains (int): 域的数量 (PACS 为 4)
            metric (str): 原型网络的距离度量 ('euclidean' 或 'cosine')
            intrinsic_encoder_drop_rate (float): 本征编码器ResNet12残差块Dropout率
            domain_dropout (float): 域分类器dropout率
            use_intrinsic_supcon (bool): 是否使用跨域监督对比学习 (本征支)
            intrinsic_supcon_weight (float): SupCon损失权重
            intrinsic_supcon_proj_dim (int): SupCon投影空间维度
            intrinsic_supcon_dropout (float): SupCon投影头dropout率
            intrinsic_supcon_phase1_temp (float): 本征支早期温度 (epoch 0-15)
            intrinsic_supcon_phase2_temp (float): 本征支中期温度 (epoch 30-70)
            intrinsic_supcon_phase3_temp (float): 本征支后期温度 (epoch 70-100)
            intrinsic_supcon_phase1_end (int): 早期结束 epoch
            intrinsic_supcon_phase2_start (int): 中期平台开始 epoch
            intrinsic_supcon_phase2_end (int): 中期平台结束 epoch
            intrinsic_supcon_cross_domain_weight (float): 跨域同类正样本权重 (强化域不变性)
            intrinsic_supcon_same_domain_weight (float): 同域同类正样本权重 (避免域捷径)
            intrinsic_supcon_cross_domain_neg_weight (float): 跨域异类负样本权重 (强化区分)
            total_epochs (int): 总训练epoch数
            use_domain_supcon (bool): 是否使用域监督对比学习 (域支)
            domain_supcon_weight (float): 域SupCon损失权重
            domain_supcon_proj_dim (int): 域SupCon投影空间维度
            domain_supcon_dropout (float): 域SupCon投影头dropout率
            domain_supcon_early_temp (float): 域SupCon早期温度
            domain_supcon_mid_temp (float): 域SupCon中期温度
            domain_supcon_final_temp (float): 域SupCon后期温度
            domain_supcon_early_epochs (int): 域SupCon早期阶段结束epoch
            domain_supcon_mid_epochs (int): 域SupCon中期阶段结束epoch
            domain_supcon_cross_class_weight (float): 同域异类正样本权重
            domain_supcon_same_class_weight (float): 同域同类正样本权重
            domain_supcon_cdsc_neg_weight (float): 域SupCon跨域同类负样本权重
            use_sap (bool): 是否使用 SAP 语义锚点净化模块
            sap_dropout (float): SAP 投影层 dropout 率
            sap_orth_weight (float): SAP 正交损失权重
            k_shot (int): 每类支持样本数 (用于 SAP 模块)
        """
        super().__init__()
        self.n_domains = n_domains
        self.use_intrinsic_supcon = use_intrinsic_supcon
        self.intrinsic_supcon_weight = intrinsic_supcon_weight
        self.metric = metric
        self.k_shot = k_shot  # 保存 k_shot 用于 SAP 模块
        
        # 构造域支配置：显式风格统计编码（禁BN）
        domain_cfg = StyleStatDomainEncoderConfig(
            domain_dim=domain_dim,
            kernel_size=domain_filterbank_kernel,
            use_gabor=domain_filterbank_use_gabor,
            use_dog=domain_filterbank_use_dog,
            trainable_gain=domain_filterbank_trainable_gain,
            trainable_delta_kernel=domain_filterbank_trainable_delta_kernel,
            token_dropout=domain_token_dropout,
            token_scale_init=domain_token_scale_init,
            use_gram_stats=domain_use_gram_stats,
            gram_rank=domain_gram_rank,
            gram_out_dim=domain_gram_out_dim,
            norm=domain_norm,
        )

        self.backbone = DualStreamResNet12(drop_rate=intrinsic_encoder_drop_rate, domain_cfg=domain_cfg)
        self.proto_head = PrototypeNetwork(metric=metric)
        
        # 域分类器: 预测域标签 (0-3) - 使用可配置的dropout
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.backbone.domain_dim, 64),
            nn.ReLU(),
            nn.Dropout(domain_dropout),
            nn.Linear(64, n_domains)
        )
        
        # 本征支监督对比学习模块 (与域支对称设计)
        # 特性: 分段温度调度 + 域感知正/负样本加权
        if use_intrinsic_supcon:
            self.intrinsic_supcon_module = IntrinsicSupConLoss(
                in_dim=self.backbone.feature_dim,  # 640
                proj_dim=intrinsic_supcon_proj_dim,
                hidden_dim=256,
                dropout=intrinsic_supcon_dropout,
                # 分段温度调度 (与域支协同)
                phase1_temp=intrinsic_supcon_phase1_temp,
                phase2_temp=intrinsic_supcon_phase2_temp,
                phase3_temp=intrinsic_supcon_phase3_temp,
                phase1_end=intrinsic_supcon_phase1_end,
                phase2_start=intrinsic_supcon_phase2_start,
                phase2_end=intrinsic_supcon_phase2_end,
                total_epochs=total_epochs,
                # 正样本权重
                cross_domain_weight=intrinsic_supcon_cross_domain_weight,
                same_domain_weight=intrinsic_supcon_same_domain_weight,
                # 负样本权重
                cross_domain_neg_weight=intrinsic_supcon_cross_domain_neg_weight
            )
        else:
            self.intrinsic_supcon_module = None
        
        # 域监督对比学习模块 (域支)
        self.use_domain_supcon = use_domain_supcon
        self.domain_supcon_weight = domain_supcon_weight
        if use_domain_supcon:
            self.domain_supcon_module = DomainSupConLoss(
                in_dim=self.backbone.domain_dim,  # domain_dim (默认 64)
                proj_dim=domain_supcon_proj_dim,
                hidden_dim=domain_supcon_proj_dim,  # hidden_dim == proj_dim（轻量瓶颈）
                dropout=domain_supcon_dropout,
                early_temp=domain_supcon_early_temp,
                mid_temp=domain_supcon_mid_temp,
                final_temp=domain_supcon_final_temp,
                early_epochs=domain_supcon_early_epochs,
                mid_epochs=domain_supcon_mid_epochs,
                total_epochs=total_epochs,
                cross_class_weight=domain_supcon_cross_class_weight,
                same_class_weight=domain_supcon_same_class_weight,
                cdsc_neg_weight=domain_supcon_cdsc_neg_weight  # 新增传递
            )
        else:
            self.domain_supcon_module = None
        
        # SAP 语义锚点净化模块
        self.use_sap = use_sap
        self.sap_orth_weight = sap_orth_weight
        if use_sap:
            self.sap_module = SemanticAnchorPurification(
                dom_dim=self.backbone.domain_dim,  # 64
                phy_dim=self.backbone.feature_dim,  # 640
                dropout=sap_dropout,
                normalize_query=True,
                use_layernorm=True
            )
        else:
            self.sap_module = None
        
        self.mode = 'train'
        self.current_epoch = 0  # 用于温度调度
        
        # 打印模型信息
        self._print_model_info()

    def _print_model_info(self):
        """打印模型配置信息"""
        # 避免 Windows 控制台在 GBK 编码下打印 emoji 导致 UnicodeEncodeError
        print("[INFO] Experiment B.1 Model 配置:")
        print(f"   骨干网络: DualStreamResNet12")
        print(f"   域数量: {self.n_domains}")
        print(f"   使用 SupCon (本征支): {self.use_intrinsic_supcon}")
        if self.use_intrinsic_supcon:
            print(f"   SupCon 权重: {self.intrinsic_supcon_weight}")
        print(f"   使用 DomainSupCon (域支): {self.use_domain_supcon}")
        if self.use_domain_supcon:
            print(f"   DomainSupCon 权重: {self.domain_supcon_weight}")
        print(f"   使用 SAP 净化模块: {self.use_sap}")
        if self.use_sap:
            print(f"   SAP 正交损失权重: {self.sap_orth_weight}")
        print(f"   距离度量: {self.metric}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")

    def extract_features(self, images):
        """
        提取图像特征 (本征特征和域特征)
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            f_phy: 本征特征 [B, 640]
            f_dom: 域特征 [B, domain_dim]
        """
        return self.backbone(images)

    def forward(self, support_images, support_labels, query_images, n_way,
                query_domain_labels=None, query_labels=None, support_domain_labels=None):
        """
        前向传播
        
        Args:
            support_images: 支持集图像
            support_labels: 支持集标签
            query_images: 查询集图像
            n_way: 类别数 (N-way)
            query_domain_labels: [n_query] 查询集的域标签 (用于计算域损失)
            query_labels: [n_query] 查询集类别标签 (用于SupCon损失)
            support_domain_labels: [n_support] 支持集的域标签 (用于SupCon跨域对比)
            
        Returns:
            logits: 原型分类 logits
            prototypes: 计算出的原型
            domain_logits: 域分类 logits
            intrinsic_supcon_loss: SupCon 对比损失 (本征支)
            intrinsic_supcon_stats: SupCon 统计信息
            domain_supcon_loss: 域 SupCon 对比损失 (域支)
            domain_supcon_stats: 域 SupCon 统计信息
            sap_loss: SAP 正交损失
            sap_stats: SAP 统计信息
        """
        # 特征提取
        s_phy, s_dom = self.extract_features(support_images)  # s_phy: [N_s, 640], s_dom: [N_s, 64]
        q_phy, q_dom = self.extract_features(query_images)    # q_phy: [N_q, 640], q_dom: [N_q, 64]
        
        # SAP 净化模块 (在使用域特征之前进行净化)
        sap_loss = torch.tensor(0.0, device=support_images.device)
        sap_stats = {}
        if self.use_sap and self.mode == 'train' and query_labels is not None:
            # 合并 support 和 query 特征
            all_phy = torch.cat([s_phy, q_phy], dim=0)  # [N_s + N_q, 640]
            all_dom = torch.cat([s_dom, q_dom], dim=0)  # [N_s + N_q, 64]
            all_labels = torch.cat([support_labels, query_labels], dim=0)  # [N_s + N_q]
            
            # 调用 SAP 模块净化域特征
            all_dom_pure, attn_weights, sap_loss, sap_stats = self.sap_module(
                f_dom=all_dom,
                f_phy=all_phy,
                labels=all_labels,
                n_way=n_way,
                k_shot=self.k_shot
            )  # all_dom_pure: [N_s + N_q, 64]
            
            # 分离 support 和 query 的净化特征
            support_size = n_way * self.k_shot
            s_dom = all_dom_pure[:support_size]  # [N_s, 64]
            q_dom = all_dom_pure[support_size:]  # [N_q, 64]
        
        # 1. 原型分类 (L_cls) - 使用 L2 归一化特征
        s_phy_norm = F.normalize(s_phy, p=2, dim=1)  # [N_s, 640]
        q_phy_norm = F.normalize(q_phy, p=2, dim=1)  # [N_q, 640]
        logits, prototypes = self.proto_head(s_phy_norm, support_labels, q_phy_norm, n_way)
        
        # 2. 域分类 (L_dom_CE) - 使用净化后的域特征
        #    改进：Support + Query 联合训练，利用 s_dom 的梯度回传增强域编码器的稳定性
        domain_logits = None
        if self.mode == 'train' and query_domain_labels is not None:
            if support_domain_labels is not None:
                # 拼接 Support + Query 净化后的域特征
                all_dom = torch.cat([s_dom, q_dom], dim=0)  # [N_s + N_q, 64]
                domain_logits = self.domain_classifier(all_dom)
            else:
                # 仅 Query (回退兼容)
                domain_logits = self.domain_classifier(q_dom)

        # 4. 本征 SupCon 对比损失 (L_phy_supcon) - 本征支，带域感知权重
        #    权重策略: 跨域同类=1.5 (强化域不变性), 同域同类=0.8 (避免域捷径)
        #    与域支保持一致的调用方式: 先合并特征再计算
        intrinsic_supcon_loss = torch.tensor(0.0, device=support_images.device)
        intrinsic_supcon_stats = {}
        if self.use_intrinsic_supcon and self.mode == 'train' and query_labels is not None:
            # 合并支持集和查询集 (与 domain_supcon 调用方式一致)
            all_phy_features = torch.cat([s_phy, q_phy], dim=0)  # [N_s + N_q, 640]
            all_class_labels = torch.cat([support_labels, query_labels], dim=0)
            
            # 合并域标签 (如果有)
            all_domain_labels = None
            if support_domain_labels is not None and query_domain_labels is not None:
                all_domain_labels = torch.cat([support_domain_labels, query_domain_labels], dim=0)
            
            intrinsic_supcon_loss, intrinsic_supcon_stats = self.intrinsic_supcon_module(
                intrinsic_features=all_phy_features,
                class_labels=all_class_labels,
                domain_labels=all_domain_labels,
                epoch=self.current_epoch
            )
        
        # 5. 域 SupCon 对比损失 (L_dom_supcon) - 使用净化后的域特征
        #    目标: 让 f_dom_pure 只编码域统计信息，压制类别可分性
        #    梯度流: domain_supcon_loss → domain_projection_head → domain_encoder
        #           (不影响 intrinsic_encoder，因为两个编码器完全独立)
        domain_supcon_loss = torch.tensor(0.0, device=support_images.device)
        domain_supcon_stats = {}
        # 必须同时有域标签和类别标签才能计算域对比损失
        has_domain_labels = (support_domain_labels is not None and query_domain_labels is not None)
        if self.use_domain_supcon and self.mode == 'train' and query_labels is not None and has_domain_labels:
            # 合并支持集和查询集: [N_s + N_q, ...]
            all_dom_features = torch.cat([s_dom, q_dom], dim=0)  # [N_s + N_q, 64] (已净化)
            all_domain_labels = torch.cat([support_domain_labels, query_domain_labels], dim=0)
            all_class_labels = torch.cat([support_labels, query_labels], dim=0)
            
            domain_supcon_loss, domain_supcon_stats = self.domain_supcon_module(
                domain_features=all_dom_features,
                domain_labels=all_domain_labels,
                class_labels=all_class_labels,
                epoch=self.current_epoch
            )

        return (logits, prototypes, domain_logits,
                intrinsic_supcon_loss, intrinsic_supcon_stats, 
                domain_supcon_loss, domain_supcon_stats,
                sap_loss, sap_stats)
    
    def set_mode(self, mode):
        """
        设置模型模式 (train/eval)
        """
        self.mode = mode
        if mode == 'train':
            self.train()
        else:
            self.eval()
    
    def set_epoch(self, epoch: int):
        """
        设置当前epoch (用于SupCon温度调度)
        
        Args:
            epoch: 当前epoch
        """
        self.current_epoch = epoch
            
    def to_device(self, device):
        """
        将模型移动到指定设备
        """
        self.to(device)
        print(f"[OK] ExpB1Model 已移动到设备: {device}")
        return self
            
    def get_parameters(self):
        """
        获取模型参数（Patch 3: 对 SAP 的 logit_scale_log 禁用 weight_decay）
        
        Returns:
            list: 参数组列表，格式为 [{'params': [...], 'weight_decay': ...}, ...]
        """
        # 收集需要禁用 weight_decay 的参数
        no_decay_params = []
        if self.use_sap and hasattr(self.sap_module, 'logit_scale_log'):
            no_decay_params.append(self.sap_module.logit_scale_log)
        
        # 收集其他所有参数
        no_decay_ids = {id(p) for p in no_decay_params}
        decay_params = [p for p in self.parameters() if id(p) not in no_decay_ids]
        
        # 返回参数组
        param_groups = [
            {'params': decay_params},  # 使用默认 weight_decay
            {'params': no_decay_params, 'weight_decay': 0.0}  # 禁用 weight_decay
        ]
        
        return param_groups
        
    def get_model_info(self):
        """
        获取模型信息字典
        """
        return {
            'model_name': 'ExpB1Model',
            'backbone': 'DualStreamResNet12',
            'use_hsic': self.use_hsic,
            'use_intrinsic_supcon': self.use_intrinsic_supcon,
            'metric': self.metric
        }
    
    def get_intrinsic_supcon_stats(self) -> dict:
        """
        获取本征支SupCon模块的当前温度
        
        Returns:
            包含温度信息的字典
        """
        if self.intrinsic_supcon_module is not None:
            temp = self.intrinsic_supcon_module.temp_scheduler.get_temperature(self.current_epoch)
            return {'temperature': temp, 'epoch': self.current_epoch}
        return {}

# 模型配置信息 - 更新为ExpB1Model完整配置
MODEL_CONFIG = {
    'model_name': 'ExpB1Model',
    'description': '双流网络 + 域分类器 + SupCon对比学习 - 用于域泛化小样本学习',
    'backbone': 'DualStreamResNet12',
    'head_type': 'prototype',
    'key_features': [
        '双流特征解耦 (本征/域分支)',
        '域分类器对抗训练',
        '跨域监督对比学习 (SupCon)',
        '动态温度调度',
        '支持欧氏距离和余弦距离'
    ]
}
