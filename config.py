import torch

class Config:
    # 数据配置 - PACS域泛化设置
    dataset = "PACS"
    n_way = 5
    k_shot = 5
    query_per_class = 15
    test_episodes = 2000
    
    # PACS域配置
    n_domains = 4  # PACS有4个域: photo, art_painting, cartoon, sketch
    source_domains = ['photo', 'art_painting', 'cartoon']  # 默认源域
    target_domain = 'sketch'  # 默认目标域
      
    # 训练配置
    num_epochs = 60  
    episodes_per_epoch = 150  # 每个 epoch 训练 170 个 episode
    learning_rate = 0.05    
    optimizer = "SGD"
    momentum = 0.9
    weight_decay = 5e-4  # 权重衰减系数（L2正则化系数）：在每次参数更新时对权重施加惩罚，防止模型过拟合；5e-4 表示对权重乘以 (1 - 5e-4 * lr) 进行衰减，值越大正则化越强
    nesterov = True  # SGD的Nesterov动量
    use_wandb = False  # 训练阶段开启，测试阶段关闭（测试脚本会显式关闭）
    
    # 验证配置
    val_episodes = 150      # 每次常规验证运行的episode数
    eval_frequency = 5      # 每n个epoch进行一次常规验证
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True
    amp_dtype = "bf16"
    
    # 学习率调度配置
    # 线性预热（Linear Warmup）参数
    warmup = True  # 是否启用预热阶段。
    warmup_epochs = 10  # 预热的 epoch 数，必须满足 0 <= warmup_epochs < num_epochs
    warmup_start_lr = 1e-3  # 预热起始学习率，必须满足 0 < warmup_start_lr < learning_rate
    scheduler = "MultiStepLR"   # 学习率调度器类型：MultiStepLR 表示“多步长衰减”，会在指定的 epoch 里程碑处将学习率乘以 scheduler_gamma
    scheduler_milestones = [40, 50]  # 在 40、50 epoch 处进行衰减，使 60 epoch 时学习率足够小
    scheduler_gamma = 0.5  # 每次衰减乘以 0.5，确保 90 epoch 时学习率降到较低值

    # 其他配置
    seed = 321  # 随机种子，确保实验可重复
    log_interval = 10   # 每n个打印一次训练日志
    
    # 梯度裁剪配置
    grad_clip_norm = 1.0  # 梯度裁剪的最大范数，确保训练稳定性
    
    # 模型保存配置
    save_best_only = True  # 只保存验证效果最好的模型


    """
    modules
    """
    # ExpB1Model模型参数
    metric = 'euclidean'  # 原型网络距离度量
    intrinsic_encoder_drop_rate = 0.2  # 本征编码器ResNet12的Dropout率
    manifold_reg_weight = 0.15  # 流形正则损失权重
    domain_loss_weight = 0.2  # 域分类损失权重
    domain_dropout = 0.3  # 域分类器dropout率

    # SupCon本征支监督对比学习参数
    use_intrinsic_supcon = True  # 是否使用跨域监督对比学习
    intrinsic_supcon_weight = 0.3  # SupCon损失权重
    intrinsic_supcon_proj_dim = 128  # SupCon投影空间维度
    intrinsic_supcon_dropout = 0.2  # SupCon投影头dropout率
    # 分段温度调度 (与域支倒U形协同)
    intrinsic_supcon_phase1_temp = 0.5   # 早期温度 (等待域支稳定)
    intrinsic_supcon_phase2_temp = 0.3   # 中期温度 (与域支对称)
    intrinsic_supcon_phase3_temp = 0.2  # 后期温度
    intrinsic_supcon_phase1_end = 10     # 早期结束 epoch
    intrinsic_supcon_phase2_start = 30   # 中期平台开始 epoch
    intrinsic_supcon_phase2_end = 50     # 中期平台结束 epoch
    # 正样本权重:
    intrinsic_supcon_cross_domain_weight = 1.5
    intrinsic_supcon_same_domain_weight = 0.7
    # 困难负样本权重: 跨域异类=1.2 (真正的困难负样本，需要强化区分)
    intrinsic_supcon_cross_domain_neg_weight = 1.0
    
    # 域监督对比学习参数 (DomainSupCon)
    use_domain_supcon = True  # 是否使用域监督对比学习
    domain_supcon_weight = 0.15  # 域 SupCon 损失权重
    domain_supcon_proj_dim = 32  # 域 SupCon 投影空间维度（domain_dim=64 时建议 32 形成瓶颈）
    domain_supcon_dropout = 0.2  # 域 SupCon 投影头 dropout 率
    # 倒 U 形温度调度
    domain_supcon_early_temp = 0.5   # 早期温度 (快速建立域聚类)
    domain_supcon_mid_temp = 0.6     # 中期温度 (保留域内多样性)
    domain_supcon_final_temp = 0.3  # 后期温度 (精细化域边界)
    domain_supcon_early_epochs = 15  # 早期阶段结束 epoch
    domain_supcon_mid_epochs = 40    # 中期阶段结束 epoch
    # 正样本权重
    domain_supcon_cross_class_weight = 1.2
    domain_supcon_same_class_weight = 0.3
    # 困难负样本权重 (跨域同类负样本加权)
    domain_supcon_cdsc_neg_weight = 1.5  # CDSC负样本权重 

    # ==========================================================================
    # SAP (Semantic Anchor-driven Purification) 模块配置
    # ==========================================================================
    use_sap = True  # 是否使用 SAP 语义锚点净化模块
    sap_dropout = 0.1  # SAP 投影层 dropout 率
    sap_orth_weight = 0.1  # SAP 正交损失权重

    # ==========================================================================
    # Domain Style/Stat Encoder (方案A: 固定/弱可学习滤波器组 + 显式统计)
    # ==========================================================================
    # 输出维度（连续工况坐标）
    domain_dim: int = 64

    # 固定滤波器组
    domain_filterbank_kernel: int = 7
    domain_filterbank_use_gabor: bool = True
    domain_filterbank_use_dog: bool = True
    domain_filterbank_trainable_gain: bool = True
    domain_filterbank_trainable_delta_kernel: bool = False  # 默认关闭，避免学回语义

    # Token head
    domain_token_dropout: float = 0.2
    domain_token_scale_init: float = 1.0

    # 显式统计（低秩 Gram）
    domain_use_gram_stats: bool = True
    domain_gram_rank: int = 16
    domain_gram_out_dim: int = 64

    # 归一化（明确禁用 BN）
    domain_norm: str = "gn"

    # ==========================================================================
    # ManifoldEvo 课程学习配置 (Curriculum Learning for Stable Metric)
    # ==========================================================================
    # 统一三阶段划分 (门控预热 + 协方差课程同步):
    #   Phase 0 (0 ~ phase1): 静默期 - Gate=0, Σ≡I, LogDet=0
    #   Phase 1 (phase1 ~ phase2): 注入期 - Gate↑, Scalar Σ, LogDet↑  
    #   Phase 2 (phase2 ~ end): 共演期 - Gate=1, Vector Σ, LogDet=final
    curriculum_phase1_epoch: int = 30        # Phase 0→1 边界 (静默期结束)
    curriculum_phase2_epoch: int = 60        # Phase 1→2 边界 (注入期结束)
    
    # Log-Sigma 参数化范围 (确保正定 + 条件数上界)
    # σ² ∈ [exp(-4), exp(2)] ≈ [0.018, 7.39], κ_max ≈ 403
    logsigma_min: float = -4.0               # log(σ²) 下限
    logsigma_max: float = 2.0                # log(σ²) 上限
    
    # LogDet 课程调度 (渐进启用，防止拉锯梯度)
    logdet_start_alpha: float = 0.0          # logdet 初始 α (Phase 0)
    logdet_final_alpha: float = 0.1          # logdet 最终 α (Phase 2)
    
    # 演化门控上限 (防止原型过度漂移)
    evo_gate_final_scale: float = 1.0        # Gate 最终上限
    
    # Evo 模块优化器配置 (敏感模块保护)
    uncertainty_lr_ratio: float = 0.3        # uncertainty_net LR 倍率
    projector_lr_ratio: float = 0.2          # projector LR 倍率
    use_adamw_evo: bool = True               # Evo 模块用 AdamW
    evo_weight_decay: float = 2e-4           # Evo 模块权重衰减

    # 打印设备信息
    @classmethod
    def print_device_info(cls):
        if torch.cuda.is_available():
            print("✅ 成功调用GPU!")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   当前设备: {cls.device}")
        else:
            print("⚠️  GPU不可用，使用CPU")
            print(f"   当前设备: {cls.device}")
