# 标准库导入
import os
import random
import argparse
import time

# 第三方库导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 本地模块导入
from config import Config
from utils.scheduler import Scheduler
from utils.index import compute_episode_accuracy, compute_confidence_interval, compute_epoch_statistics, compute_prototype_separation_ratio
from utils.dataloader_improvement import PACSDataset, create_cross_domain_episode_loader, get_pacs_transform  # 更新为改进版dataloader
from utils.visualization import (visualize_alpha_weights, plot_epoch_accuracy, plot_epoch_statistics, 
                                plot_training_curve, plot_accuracy_comparison, plot_accuracy_heatmap, plot_val_accuracy_curve, plot_separation_ratio_curve,
                                plot_tsne_embeddings, plot_tsne_evolution, plot_leakindex_curve)
from utils.leakindex import compute_leak_index
from utils.tsne_manager import TSNEVisualizer


def setup_environment(seed):
    """
    设置环境种子以确保实验可重复性
    
    Args:
        seed (int): 随机种子值
        
    Note:
        - 设置PyTorch、NumPy、Python random的种子
        - 配置CUDNN为确定性模式（可能影响性能但保证可重复性）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置cudnn确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def create_model(config):
    """
    创建ExpB1Model双流架构并部署到GPU
    
    Args:
        config: 配置对象，包含模型超参数
        
    Returns:
        ExpB1Model: 实验B.1双流模型实例
    """
    from model.exp_b1_model import ExpB1Model
    
    # 创建ExpB1Model实例并直接部署到GPU
    model = ExpB1Model(
        n_domains=config.n_domains,  # PACS有4个域
        metric=config.metric,         # 距离度量方式
        intrinsic_encoder_drop_rate=config.intrinsic_encoder_drop_rate,
        # 本征支对比学习参数 (分段温度调度)
        use_intrinsic_supcon=config.use_intrinsic_supcon,
        intrinsic_supcon_weight=config.intrinsic_supcon_weight,
        intrinsic_supcon_proj_dim=config.intrinsic_supcon_proj_dim,
        intrinsic_supcon_dropout=config.intrinsic_supcon_dropout,
        intrinsic_supcon_phase1_temp=config.intrinsic_supcon_phase1_temp,
        intrinsic_supcon_phase2_temp=config.intrinsic_supcon_phase2_temp,
        intrinsic_supcon_phase3_temp=config.intrinsic_supcon_phase3_temp,
        intrinsic_supcon_phase1_end=config.intrinsic_supcon_phase1_end,
        intrinsic_supcon_phase2_start=config.intrinsic_supcon_phase2_start,
        intrinsic_supcon_phase2_end=config.intrinsic_supcon_phase2_end,
        intrinsic_supcon_cross_domain_weight=config.intrinsic_supcon_cross_domain_weight,
        intrinsic_supcon_same_domain_weight=config.intrinsic_supcon_same_domain_weight,
        intrinsic_supcon_cross_domain_neg_weight=config.intrinsic_supcon_cross_domain_neg_weight,
        total_epochs=config.num_epochs,
        # 域支对比学习参数 (倒U形温度调度)
        use_domain_supcon=config.use_domain_supcon,
        domain_supcon_weight=config.domain_supcon_weight,
        domain_supcon_proj_dim=config.domain_supcon_proj_dim,
        domain_supcon_dropout=config.domain_supcon_dropout,
        domain_supcon_early_temp=config.domain_supcon_early_temp,
        domain_supcon_mid_temp=config.domain_supcon_mid_temp,
        domain_supcon_final_temp=config.domain_supcon_final_temp,
        domain_supcon_early_epochs=config.domain_supcon_early_epochs,
        domain_supcon_mid_epochs=config.domain_supcon_mid_epochs,
        domain_supcon_cross_class_weight=config.domain_supcon_cross_class_weight,
        domain_supcon_same_class_weight=config.domain_supcon_same_class_weight,
        domain_supcon_cdsc_neg_weight=config.domain_supcon_cdsc_neg_weight,

        # ==========================
        # 域支风格统计编码器（方案A）
        # ==========================
        domain_dim=config.domain_dim,
        domain_filterbank_kernel=config.domain_filterbank_kernel,
        domain_filterbank_use_gabor=config.domain_filterbank_use_gabor,
        domain_filterbank_use_dog=config.domain_filterbank_use_dog,
        domain_filterbank_trainable_gain=config.domain_filterbank_trainable_gain,
        domain_filterbank_trainable_delta_kernel=config.domain_filterbank_trainable_delta_kernel,
        domain_token_dropout=config.domain_token_dropout,
        domain_token_scale_init=config.domain_token_scale_init,
        domain_use_gram_stats=config.domain_use_gram_stats,
        domain_gram_rank=config.domain_gram_rank,
        domain_gram_out_dim=config.domain_gram_out_dim,
        domain_norm=config.domain_norm,
        # ==========================
        # SAP 模块配置
        # ==========================
        use_sap=config.use_sap,
        sap_dropout=config.sap_dropout,
        sap_orth_weight=config.sap_orth_weight,
        k_shot=config.k_shot,
    )
    
    # 部署到设备
    device = Config.device
    model = model.to(device=device)
    
    # 应用channels_last内存格式优化（可提升20-30%性能）
    channels_last_enabled = False
    if torch.cuda.is_available():
        try:
            model = model.to(memory_format=torch.channels_last)
            channels_last_enabled = True
            print("✅ 模型已转换为channels_last内存格式 (预期性能提升20-30%)")
        except RuntimeError as e:
            print(f"⚠️  channels_last转换失败: {e}")
            print("   继续使用默认contiguous格式")
    else:
        print("⚠️  CPU模式不支持channels_last优化")
    
    print(f"📋 已创建模型: {model.__class__.__name__}")
    print(f"🚀 模型已部署到: {device}")
    print(f"🔧 模型配置: n_domains={config.n_domains}, metric={config.metric}")
    print(f"🔗 本征支SupCon: use={config.use_intrinsic_supcon}, weight={config.intrinsic_supcon_weight}")
    print(f"🔗 域支DomSupCon: use={config.use_domain_supcon}, weight={config.domain_supcon_weight}")
    print(f"🔗 SAP净化模块: use={config.use_sap}, orth_weight={config.sap_orth_weight}")
    
    return model, channels_last_enabled


def run_episode(model, support_images, support_labels, query_images, n_way, 
                query_domain_labels=None, query_labels=None, support_domain_labels=None):
    """
    运行单个episode - ExpB1Model双流架构 + 双向对比学习 + SAP净化
    
    Args:
        model: ExpB1Model实例
        support_images: 支持集图像 [B, C, H, W]
        support_labels: 支持集标签 [B]
        query_images: 查询集图像 [B, C, H, W]
        n_way: 类别数
        query_domain_labels: 查询集域标签（训练时使用）
        query_labels: 查询集类别标签（用于SupCon损失）
        support_domain_labels: 支持集域标签（用于SupCon跨域权重）
        
    Returns:
        tuple: (logits, prototypes, domain_logits,
                intrinsic_supcon_loss, intrinsic_supcon_stats, 
                domain_supcon_loss, domain_supcon_stats,
                sap_loss, sap_stats)

    Note:
        双流架构返回: 分类logits, 原型, 域分类logits,
                     本征支SupCon损失, 本征支SupCon统计, 
                     域支SupCon损失, 域支SupCon统计,
                     SAP正交损失, SAP统计
    """
    return model(support_images, support_labels, query_images, n_way, 
                 query_domain_labels, query_labels, support_domain_labels)


# 使用 utils.index 中的 compute_epoch_statistics，移除本地重复实现


def evaluate_model(model, dataset, config, num_test_episodes=100):
    """
    评估模型性能
    """
    # 设置为评估模式
    model.set_mode('eval')
    
    accuracies = []
    
    # 创建测试episode加载器 - 使用跨域采样
    # 注意：这里我们使用 args.test_source_domains 和 args.test_query_domains
    # 但 evaluate_model 函数签名没有这些参数，我们需要从 config 获取
    episode_loader = create_cross_domain_episode_loader(
        dataset, config.n_way, config.k_shot, 
        config.query_per_class, num_test_episodes,
        support_domain_pool=config.test_source_domains,
        query_domain_pool=config.test_query_domains
    )
    
    use_amp = getattr(Config, 'use_amp', False) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if getattr(Config, 'amp_dtype', 'bf16').lower() == 'bf16' else torch.float16
    with torch.no_grad():
        for episode_idx, episode_data in enumerate(episode_loader):
            if len(episode_data) == 6:
                support_images, support_labels, query_images, query_labels, _, _ = episode_data
            else:
                support_images, support_labels, query_images, query_labels = episode_data[:4]
            # 移动到设备
            support_images = support_images.to(config.device)
            support_labels = support_labels.to(config.device)
            query_images = query_images.to(config.device)
            query_labels = query_labels.to(config.device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = run_episode(
                        model, support_images, support_labels,
                        query_images, config.n_way, query_domain_labels=None
                    )[0]
            else:
                logits = run_episode(
                    model, support_images, support_labels,
                    query_images, config.n_way, query_domain_labels=None
                )[0]
            
            # 计算准确率
            acc = compute_episode_accuracy(logits, query_labels)
            accuracies.append(acc)
            
            if (episode_idx + 1) % 10 == 0:
                print(f"  Evaluated {episode_idx + 1}/{num_test_episodes} episodes")
    
    # 计算平均准确率和置信区间
    mean_acc = np.mean(accuracies)
    lower_bound, upper_bound = compute_confidence_interval(accuracies)
    
    # 重置为训练模式
    model.set_mode('train')
    
    return mean_acc, lower_bound, upper_bound


def main():
    """
    主训练函数
    """
    # 解析命令行参数 - 更新为PACS数据集路径
    parser = argparse.ArgumentParser(description='Train ExpB1Model - Dual Stream + HSIC for Domain Generalization (Improved Sampling)')
    parser.add_argument('--pacs_root', type=str, required=True, help='Path to PACS dataset root directory')
    
    # 训练阶段域配置
    parser.add_argument('--train_source_domains', nargs='+', default=['photo', 'art_painting', 'cartoon'], 
                       help='Source domains for training support set')
    parser.add_argument('--train_query_domains', nargs='+', default=['photo', 'art_painting', 'cartoon'], 
                       help='Domains for training query set (will be sampled consistently per episode)')
                       
    # 验证阶段域配置
    parser.add_argument('--test_source_domains', nargs='+', default=['photo', 'art_painting', 'cartoon'], 
                       help='Source domains for validation support set')
    parser.add_argument('--test_query_domains', nargs='+', default=['sketch'], 
                       help='Target domain for validation query set')
                       
    parser.add_argument('--num_epochs', type=int, default=Config.num_epochs, help='Number of epochs to train')
    parser.add_argument('--episodes_per_epoch', type=int, default=Config.episodes_per_epoch, help='Number of episodes per epoch')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.num_epochs = args.num_epochs
    Config.episodes_per_epoch = args.episodes_per_epoch
    
    # 将参数注入 Config 以便在 evaluate_model 中使用
    Config.train_source_domains = args.train_source_domains
    Config.train_query_domains = args.train_query_domains
    Config.test_source_domains = args.test_source_domains
    Config.test_query_domains = args.test_query_domains
    
    # 设置环境
    setup_environment(Config.seed)
    
    # 打印设备信息
    Config.print_device_info()
    
    # 创建模型（已包含GPU部署）- 使用完整配置
    model, channels_last_enabled = create_model(Config)
    
    # 获取模型参数组（Patch 3: 支持参数分组）
    param_groups = model.get_parameters()
    
    # 优化器配置 - 使用配置参数
    optimizer = torch.optim.SGD(param_groups, lr=Config.learning_rate, 
                                momentum=Config.momentum, weight_decay=Config.weight_decay,
                                nesterov=Config.nesterov)
    
    # 提取所有参数用于梯度裁剪（从参数组中展开）
    all_params = []
    for group in param_groups:
        all_params.extend(group['params'])
    
    # 组合调度器：线性预热 + MultiStepLR
    scheduler = Scheduler(optimizer)
    
    # 损失函数配置
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    use_amp = getattr(Config, 'use_amp', False) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if getattr(Config, 'amp_dtype', 'bf16').lower() == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    
    print(f"🔧 训练配置:")
    print(f"   学习率: {Config.learning_rate} (预热: {Config.warmup_epochs} epochs)")
    print(f"   损失权重 - Domain_loss: {Config.domain_loss_weight}")
    print(f"   梯度裁剪: {Config.grad_clip_norm}")
    print(f"   验证频率: 每 {Config.eval_frequency} epochs")
    print(f"   AMP: {use_amp} (dtype={getattr(Config, 'amp_dtype', 'bf16')})")
    
    # 数据预处理 - 使用PACS专用transform
    train_transform = get_pacs_transform(image_size=84, split='train')
    eval_transform = get_pacs_transform(image_size=84, split='test')
    
    # 加载PACS数据集 - 域泛化设置
    # 训练集需要包含所有训练阶段用到的域
    train_domains = list(set(args.train_source_domains + args.train_query_domains))
    train_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=train_domains,
        split='train',
        transform=train_transform
    )
    
    # 验证集需要包含所有验证阶段用到的域
    val_domains = list(set(args.test_source_domains + args.test_query_domains))
    val_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=val_domains,
        split='test',  # PACS没有独立的val集，使用test作为验证
        transform=eval_transform
    )

    # T-SNE 专用固定数据集 (使用eval_transform避免随机增强，确保可视化一致性)
    # 注意：为了对齐域泛化评估（source→target），需要包含test_source_domains和test_query_domains
    tsne_domains = list(set(args.test_source_domains + args.test_query_domains))
    tsne_dataset = PACSDataset(
        root_dir=args.pacs_root,
        target_domains=tsne_domains,  # 使用验证阶段的域（source + target）
        split='train',
        transform=eval_transform      # 关键：使用评估时的变换（无随机增强）
    )
    
    print(f"📊 数据集配置 (Improved):")
    print(f"   训练支持域: {args.train_source_domains}")
    print(f"   训练查询域: {args.train_query_domains}")
    print(f"   验证支持域: {args.test_source_domains}")
    print(f"   验证查询域: {args.test_query_domains}")
    print(f"   训练样本: {len(train_dataset)} 张图像 (涵盖 {train_domains})")
    print(f"   验证样本: {len(val_dataset)} 张图像 (涵盖 {val_domains})")
    
    # 训练历史记录
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    epoch_stds = []  # 用于记录每个epoch的标准差
    
    # 热力图数据收集：不同阶段的准确率矩阵
    heatmap_data = []
    sep_ratio_curve = []
    sep_ratio_epochs = []
    best_val_acc = 0.0
    
    # T-SNE可视化数据收集
    tsne_metrics_history = []
    tsne_epochs = []
    tsne_save_dir = "figures/T-SNE"
    os.makedirs(tsne_save_dir, exist_ok=True)
    print(f"📊 T-SNE可视化将保存到: {tsne_save_dir} (每5个epoch)")

    # 初始化固定Episode的T-SNE可视化器
    # 对齐域泛化评估设置：support来自source域，query来自target域（sketch）
    tsne_visualizer = TSNEVisualizer(
        dataset=tsne_dataset,
        n_way=Config.n_way,
        k_shot=Config.k_shot,
        query_per_class=Config.query_per_class,
        num_episodes=3,  # 固定3个episode用于可视化
        support_domain_pool=Config.test_source_domains,  # 使用验证阶段的source域
        query_domain_pool=Config.test_query_domains,      # 使用验证阶段的target域（sketch）
        device=Config.device
    )
    
    # ===== 对比学习统计指标历史记录 =====
    # 本征支指标
    intrinsic_pos_sim_history = []      # 同类正样本相似度
    intrinsic_neg_sim_history = []      # 异类负样本相似度
    intrinsic_cross_domain_sim_history = []  # 跨域同类相似度
    # 域支指标
    domain_cross_class_sim_history = []  # SDDC: 同域异类 (核心)
    domain_same_class_sim_history = []   # SDSC: 同域同类
    domain_cdsc_sim_history = []         # CDSC: 跨域同类 (泄露检测)
    domain_cddc_sim_history = []         # CDDC: 跨域异类 (纯负样本)
    
    # ===== LeakIndex 历史记录 (新增) =====
    leakindex_history = []               # LeakIndex = CDSC - CDDC (带符号)
    leakintensity_history = []           # LeakIntensity = max(0, LeakIndex) (仅正值)
    


    print("Starting training...")
    
    # 检查eval_frequency参数的有效性
    if not isinstance(Config.eval_frequency, int) or Config.eval_frequency < 1:
        raise ValueError("eval_frequency must be a positive integer")
    
    # 记录总训练开始时间
    total_start_time = time.time()
    epoch_times = []  # 记录每个epoch的时间
    
    # 训练循环
    for epoch in tqdm(range(Config.num_epochs), desc="Training Progress", unit="epoch"):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        # 在每个 epoch 开始时更新学习率（满足预热阶段线性增长与主阶段里程碑下降的要求）
        scheduler.step()
        print(f"  Current LR: {scheduler.get_lr():.6f}")
        
        # 创建训练episode加载器 - 使用改进的跨域采样策略
        episode_loader = create_cross_domain_episode_loader(
            train_dataset, Config.n_way, Config.k_shot, 
            Config.query_per_class, Config.episodes_per_epoch,
            support_domain_pool=args.train_source_domains,
            query_domain_pool=args.train_query_domains
        )
        
        epoch_losses = []
        epoch_accuracies = []
        epoch_domain_losses = []  # 记录域分类损失
        epoch_domain_accuracies = [] # 记录域分类准确率
        epoch_intrinsic_supcon_losses = []  # 记录本征支SupCon损失
        epoch_domain_supcon_losses = []  # 记录域支SupCon损失
        epoch_cross_domain_sim = []  # 记录跨域正样本相似度
        episode_times = []
        
        # ===== 对比学习统计指标 (epoch级别) =====
        # 本征支指标
        epoch_intrinsic_pos_sim = []     # 同类正样本相似度
        epoch_intrinsic_neg_sim = []     # 异类负样本相似度
        # 域支指标
        epoch_domain_cross_class_sim = []  # SDDC（同域异类相似度，核心反映类间分离度，应越高越好）
        epoch_domain_same_class_sim = []   # SDSC（同域同类相似度，衡量类内聚合性，应越高越好）
        epoch_domain_cdsc_sim = []         # CDSC（跨域同类相似度，关键指标，监控类别信息是否泄露，应越低越好，与CDDC接近）
        epoch_domain_cddc_sim = []         # CDDC（跨域异类相似度，纯负样本，理想情况下应越低越好）
        # LeakIndex 指标 (新增)
        epoch_leak_list = []               # 每个 episode 的 LeakIndex
        # SAP 统计指标 (新增)
        epoch_sap_gate = []                # SAP 净化强度
        epoch_sap_attn_max = []            # SAP 最大注意力均值
        epoch_sap_purif_ratio = []         # SAP 净化比例
        epoch_sap_w_orth_norm = []         # SAP W_orth 权重范数
        layer_weights = None
        epoch_sep_ratios = [] if (epoch + 1) % 5 == 0 else None
        
        # 设置当前epoch用于SupCon温度调度
        model.set_epoch(epoch)
        
        # 遍历所有episodes
        episode_loader_with_progress = tqdm(episode_loader, 
                                          total=Config.episodes_per_epoch,
                                          desc=f"Epoch {epoch+1} Episodes", 
                                          leave=False, 
                                          unit="episode")
        
        for episode_idx, episode_data in enumerate(episode_loader_with_progress):
            # 处理不同长度的episode数据（兼容域标签）
            if len(episode_data) == 6:
                support_images, support_labels, query_images, query_labels, support_domains, query_domains = episode_data
                query_domain_labels = query_domains.to(Config.device)
                support_domain_labels = support_domains.to(Config.device)  # 新增: 支持集域标签
            else:
                support_images, support_labels, query_images, query_labels = episode_data[:4]
                query_domain_labels = None  # 没有域标签时使用None
                support_domain_labels = None
            
            # 记录episode开始时间
            episode_start_time = time.time()
            
            # 移动到设备并应用channels_last格式（若已启用）
            if channels_last_enabled:
                support_images = support_images.to(Config.device, memory_format=torch.channels_last)
                query_images = query_images.to(Config.device, memory_format=torch.channels_last)
            else:
                support_images = support_images.to(Config.device)
                query_images = query_images.to(Config.device)
            
            support_labels = support_labels.to(Config.device)
            query_labels = query_labels.to(Config.device)

            if support_images.dim() != 4 or query_images.dim() != 4:
                raise ValueError(f"Expected 4D images, got support {support_images.dim()}D, query {query_images.dim()}D")
            
            # 前向与损失计算（AMP）
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    (logits, prototypes, domain_logits,
                     intrinsic_supcon_loss, intrinsic_supcon_stats, 
                     domain_supcon_loss, domain_supcon_stats,
                     sap_loss, sap_stats) = run_episode(
                        model, support_images, support_labels,
                        query_images, Config.n_way, query_domain_labels, query_labels, support_domain_labels
                    )
                    cls_loss = criterion(logits, query_labels)
                    domain_loss = torch.zeros((), device=Config.device, dtype=logits.dtype)
                    domain_acc = 0.0
                    if domain_logits is not None and query_domain_labels is not None:
                        # 检查 logits 是否包含 Support+Query (N_s + N_q)
                        if domain_logits.size(0) > query_domain_labels.size(0):
                            if support_domain_labels is not None:
                                all_domain_labels = torch.cat([support_domain_labels, query_domain_labels], dim=0)
                                domain_loss = domain_criterion(domain_logits, all_domain_labels)
                                _, domain_preds = torch.max(domain_logits, 1)
                                domain_acc = (domain_preds == all_domain_labels).float().mean().item()
                            else:
                                # 异常情况：Logits 变大了但没有 Support 标签，回退到切片
                                # (理论上 forward 内部控制了，这里做防御性编程)
                                q_len = query_domain_labels.size(0)
                                domain_loss = domain_criterion(domain_logits[-q_len:], query_domain_labels)
                                _, domain_preds = torch.max(domain_logits[-q_len:], 1)
                                domain_acc = (domain_preds == query_domain_labels).float().mean().item()
                        else:
                            # 仅 Query
                            domain_loss = domain_criterion(domain_logits, query_domain_labels)
                            _, domain_preds = torch.max(domain_logits, 1)
                            domain_acc = (domain_preds == query_domain_labels).float().mean().item()
                    
                    # 计算总损失 (包含双向对比学习 + SAP正交损失)
                    intrinsic_supcon_weight = Config.intrinsic_supcon_weight
                    domain_supcon_weight = Config.domain_supcon_weight
                    sap_orth_weight = Config.sap_orth_weight
                    total_loss = (
                        cls_loss +
                        Config.domain_loss_weight * domain_loss +
                        intrinsic_supcon_weight * intrinsic_supcon_loss +
                        domain_supcon_weight * domain_supcon_loss +
                        sap_orth_weight * sap_loss
                    )
            else:
                (logits, prototypes, domain_logits,
                 intrinsic_supcon_loss, intrinsic_supcon_stats, 
                 domain_supcon_loss, domain_supcon_stats,
                 sap_loss, sap_stats) = run_episode(
                    model, support_images, support_labels,
                    query_images, Config.n_way, query_domain_labels, query_labels, support_domain_labels
                )
                cls_loss = criterion(logits, query_labels)
                domain_loss = torch.zeros((), device=Config.device, dtype=logits.dtype)
                domain_acc = 0.0
                if domain_logits is not None and query_domain_labels is not None:
                    # 检查 logits 是否包含 Support+Query (N_s + N_q)
                    if domain_logits.size(0) > query_domain_labels.size(0):
                        if support_domain_labels is not None:
                            all_domain_labels = torch.cat([support_domain_labels, query_domain_labels], dim=0)
                            domain_loss = domain_criterion(domain_logits, all_domain_labels)
                            _, domain_preds = torch.max(domain_logits, 1)
                            domain_acc = (domain_preds == all_domain_labels).float().mean().item()
                        else:
                            # 异常情况：Logits 变大了但没有 Support 标签，回退到切片
                            q_len = query_domain_labels.size(0)
                            domain_loss = domain_criterion(domain_logits[-q_len:], query_domain_labels)
                            _, domain_preds = torch.max(domain_logits[-q_len:], 1)
                            domain_acc = (domain_preds == query_domain_labels).float().mean().item()
                    else:
                        # 仅 Query
                        domain_loss = domain_criterion(domain_logits, query_domain_labels)
                        _, domain_preds = torch.max(domain_logits, 1)
                        domain_acc = (domain_preds == query_domain_labels).float().mean().item()
                
                # 计算总损失 (包含双向对比学习 + SAP正交损失)
                intrinsic_supcon_weight = Config.intrinsic_supcon_weight
                domain_supcon_weight = Config.domain_supcon_weight
                sap_orth_weight = Config.sap_orth_weight
                total_loss = (
                    cls_loss +
                    Config.domain_loss_weight * domain_loss +
                    intrinsic_supcon_weight * intrinsic_supcon_loss +
                    domain_supcon_weight * domain_supcon_loss +
                    sap_orth_weight * sap_loss
                )

            # 增强的NaN检测：检查所有损失组件并提供详细诊断信息
            if (torch.isnan(cls_loss) or torch.isnan(domain_loss) or
                torch.isnan(intrinsic_supcon_loss) or torch.isnan(domain_supcon_loss) or
                torch.isnan(sap_loss)):
                print(f"\n❌ NaN损失检测到在Epoch {epoch+1}, Episode {episode_idx+1}:")
                print(f"   分类损失(cls_loss): {cls_loss.item()}")
                print(f"   域分类损失(domain_loss): {domain_loss.item()}")
                print(f"   本征SupCon损失(intrinsic_supcon_loss): {intrinsic_supcon_loss.item()}")
                print(f"   域SupCon损失(domain_supcon_loss): {domain_supcon_loss.item()}")
                print(f"   SAP正交损失(sap_loss): {sap_loss.item()}")
                print(f"   当前学习率: {scheduler.get_lr():.6f}")
                raise ValueError("损失组件包含NaN，训练终止")
            
            if torch.isnan(total_loss):
                raise ValueError(f"总损失为NaN (total_loss={total_loss.item()})")

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=Config.grad_clip_norm)
                
                # 梯度监控：检测梯度爆炸
                total_grad_norm = 0.0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # 警告：梯度异常大（超过10倍阈值）
                if total_grad_norm > Config.grad_clip_norm * 10:
                    print(f"\n⚠️  梯度异常大警告 - Epoch {epoch+1}, Episode {episode_idx+1}:")
                    print(f"   总梯度范数: {total_grad_norm:.2f} (阈值: {Config.grad_clip_norm})")
                    print(f"   当前学习率: {scheduler.get_lr():.6f}")
                
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=Config.grad_clip_norm)
                
                # 梯度监控：检测梯度爆炸（非AMP路径）
                total_grad_norm = 0.0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                if total_grad_norm > Config.grad_clip_norm * 10:
                    print(f"\n⚠️  梯度异常大警告 - Epoch {epoch+1}, Episode {episode_idx+1}:")
                    print(f"   总梯度范数: {total_grad_norm:.2f} (阈值: {Config.grad_clip_norm})")
                    print(f"   当前学习率: {scheduler.get_lr():.6f}")
                
                optimizer.step()
            
            # 计算准确率
            acc = compute_episode_accuracy(logits, query_labels)
            epoch_losses.append(total_loss.item())
            epoch_accuracies.append(acc)
            epoch_domain_losses.append(domain_loss.item())
            epoch_domain_accuracies.append(domain_acc)
            epoch_intrinsic_supcon_losses.append(intrinsic_supcon_loss.item())
            epoch_domain_supcon_losses.append(domain_supcon_loss.item())
            
            # ===== 记录本征支统计指标 =====
            if intrinsic_supcon_stats:
                # 跨域同类相似度 (核心指标，对应 cross_domain_weight=1.5 的正样本)
                if 'avg_cross_domain_sim' in intrinsic_supcon_stats:
                    epoch_cross_domain_sim.append(intrinsic_supcon_stats['avg_cross_domain_sim'])
                # 同域同类相似度 (对应 same_domain_weight=0.8 的正样本)
                if 'avg_same_domain_sim' in intrinsic_supcon_stats:
                    epoch_intrinsic_pos_sim.append(intrinsic_supcon_stats['avg_same_domain_sim'])
                # 负样本相似度
                if 'avg_neg_sim' in intrinsic_supcon_stats:
                    epoch_intrinsic_neg_sim.append(intrinsic_supcon_stats['avg_neg_sim'])
            
            # ===== 记录域支统计指标 =====
            if domain_supcon_stats:
                if 'avg_cross_class_sim' in domain_supcon_stats:
                    epoch_domain_cross_class_sim.append(domain_supcon_stats['avg_cross_class_sim'])
                if 'avg_same_class_sim' in domain_supcon_stats:
                    epoch_domain_same_class_sim.append(domain_supcon_stats['avg_same_class_sim'])
                if 'avg_cdsc_sim' in domain_supcon_stats:
                    epoch_domain_cdsc_sim.append(domain_supcon_stats['avg_cdsc_sim'])
                if 'avg_cddc_sim' in domain_supcon_stats:
                    epoch_domain_cddc_sim.append(domain_supcon_stats['avg_cddc_sim'])
                
                # ===== 计算并记录 LeakIndex (新增) =====
                # 只有同时存在 CDSC 和 CDDC 时才计算
                if 'avg_cdsc_sim' in domain_supcon_stats and 'avg_cddc_sim' in domain_supcon_stats:
                    leak_result = compute_leak_index(
                        cdsc=domain_supcon_stats['avg_cdsc_sim'],
                        cddc=domain_supcon_stats['avg_cddc_sim'],
                        mode='raw_diff'
                    )
                    epoch_leak_list.append(leak_result['leak_index'])
                else:
                    # 缺少必要数据时记录 NaN
                    epoch_leak_list.append(np.nan)
            
            # ===== 记录 SAP 统计指标 (新增) =====
            if sap_stats:
                if 'gate' in sap_stats:
                    epoch_sap_gate.append(sap_stats['gate'])
                if 'attn_max' in sap_stats:
                    epoch_sap_attn_max.append(sap_stats['attn_max'])
                if 'purification_ratio' in sap_stats:
                    epoch_sap_purif_ratio.append(sap_stats['purification_ratio'])
                if 'w_orth_norm' in sap_stats:
                    epoch_sap_w_orth_norm.append(sap_stats['w_orth_norm'])
            
            # 记录episode时间
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)

            if epoch_sep_ratios is not None:
                support_features, _ = model.extract_features(support_images)
                # support_features is already [B, 640] flattened vector from DualStreamResNet12
                sep_metrics = compute_prototype_separation_ratio(support_features, support_labels, prototypes)
                if 'separation_ratio' in sep_metrics:
                    r = sep_metrics['separation_ratio']
                    epoch_sep_ratios.append(float(r))
            

            
            # 打印进度 - 包含多损失信息 (SupCon + SAP)
            if (episode_idx + 1) % Config.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-Config.log_interval:])
                avg_acc = np.mean(epoch_accuracies[-Config.log_interval:])
                avg_domain = np.mean(epoch_domain_losses[-Config.log_interval:])
                avg_domain_acc = np.mean(epoch_domain_accuracies[-Config.log_interval:])
                avg_intrinsic_supcon = np.mean(epoch_intrinsic_supcon_losses[-Config.log_interval:])
                avg_domain_supcon = np.mean(epoch_domain_supcon_losses[-Config.log_interval:])
                avg_time = np.mean(episode_times[-Config.log_interval:])
                
                # 本征支相似度统计
                avg_int_cross_sim = np.mean(epoch_cross_domain_sim[-Config.log_interval:]) if epoch_cross_domain_sim else 0.0
                avg_int_pos_sim = np.mean(epoch_intrinsic_pos_sim[-Config.log_interval:]) if epoch_intrinsic_pos_sim else 0.0
                avg_int_neg_sim = np.mean(epoch_intrinsic_neg_sim[-Config.log_interval:]) if epoch_intrinsic_neg_sim else 0.0
                
                # 域支相似度统计
                avg_dom_cdsc = np.mean(epoch_domain_cdsc_sim[-Config.log_interval:]) if epoch_domain_cdsc_sim else 0.0
                avg_dom_sddc = np.mean(epoch_domain_cross_class_sim[-Config.log_interval:]) if epoch_domain_cross_class_sim else 0.0
                avg_dom_sdsc = np.mean(epoch_domain_same_class_sim[-Config.log_interval:]) if epoch_domain_same_class_sim else 0.0
                
                # SAP 统计信息
                avg_sap_gate = np.mean(epoch_sap_gate[-Config.log_interval:]) if epoch_sap_gate else 0.0
                avg_sap_attn_max = np.mean(epoch_sap_attn_max[-Config.log_interval:]) if epoch_sap_attn_max else 0.0
                avg_sap_purif_ratio = np.mean(epoch_sap_purif_ratio[-Config.log_interval:]) if epoch_sap_purif_ratio else 0.0
                avg_sap_w_orth_norm = np.mean(epoch_sap_w_orth_norm[-Config.log_interval:]) if epoch_sap_w_orth_norm else 0.0

                print(f"  Episode {episode_idx+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
                print(f"    本征支 - IntSupCon={avg_intrinsic_supcon:.4f}, 跨域同类={avg_int_cross_sim:.3f}, 同域同类={avg_int_pos_sim:.3f}, 异类={avg_int_neg_sim:.3f}")
                print(f"    域支   - DomSupCon={avg_domain_supcon:.4f}, CDSC跨域同类={avg_dom_cdsc:.3f}, SDDC同域异类={avg_dom_sddc:.3f}, SDSC同域同类={avg_dom_sdsc:.3f}")
                print(f"    SAP净化 - Gate={avg_sap_gate:.3f}, AttnMax={avg_sap_attn_max:.3f}, PurifRatio={avg_sap_purif_ratio:.3f}, W_orth={avg_sap_w_orth_norm:.2f}")
                print(f"    Time={avg_time:.2f}s")
                
        
        # 计算epoch统计信息 - 包含多损失组件 (双向对比学习)
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_acc = np.mean(epoch_accuracies)
        avg_epoch_domain = np.mean(epoch_domain_losses)
        avg_epoch_domain_acc = np.mean(epoch_domain_accuracies)
        avg_epoch_intrinsic_supcon = np.mean(epoch_intrinsic_supcon_losses)
        avg_epoch_domain_supcon = np.mean(epoch_domain_supcon_losses)
        avg_epoch_cross_sim = np.mean(epoch_cross_domain_sim) if epoch_cross_domain_sim else 0.0
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_acc)
        
        # ===== 计算对比学习统计指标均值 =====
        # 本征支指标
        avg_intrinsic_pos_sim = np.mean(epoch_intrinsic_pos_sim) if epoch_intrinsic_pos_sim else 0.0
        avg_intrinsic_neg_sim = np.mean(epoch_intrinsic_neg_sim) if epoch_intrinsic_neg_sim else 0.0
        # 域支指标
        avg_domain_cross_class_sim = np.mean(epoch_domain_cross_class_sim) if epoch_domain_cross_class_sim else 0.0
        avg_domain_same_class_sim = np.mean(epoch_domain_same_class_sim) if epoch_domain_same_class_sim else 0.0
        avg_domain_cdsc_sim = np.mean(epoch_domain_cdsc_sim) if epoch_domain_cdsc_sim else 0.0
        avg_domain_cddc_sim = np.mean(epoch_domain_cddc_sim) if epoch_domain_cddc_sim else 0.0
        
        # 添加到全局历史记录
        intrinsic_pos_sim_history.append(avg_intrinsic_pos_sim)
        intrinsic_neg_sim_history.append(avg_intrinsic_neg_sim)
        intrinsic_cross_domain_sim_history.append(avg_epoch_cross_sim)
        domain_cross_class_sim_history.append(avg_domain_cross_class_sim)
        domain_same_class_sim_history.append(avg_domain_same_class_sim)
        domain_cdsc_sim_history.append(avg_domain_cdsc_sim)
        domain_cddc_sim_history.append(avg_domain_cddc_sim)
        
        # ===== 计算并记录 LeakIndex (epoch 级别) =====
        if len(epoch_leak_list) > 0:
            # 使用 nanmean 处理可能的 NaN 值
            avg_leak = float(np.nanmean(epoch_leak_list))
            std_leak = float(np.nanstd(epoch_leak_list))
            # LeakIntensity = max(0, LeakIndex)
            avg_intensity = float(np.nanmean([max(0.0, x) for x in epoch_leak_list if not np.isnan(x)]))
        else:
            # 整个 epoch 没有有效的 LeakIndex 数据
            avg_leak = np.nan
            std_leak = np.nan
            avg_intensity = np.nan
        
        leakindex_history.append(avg_leak)
        leakintensity_history.append(avg_intensity)
        
        # 计算并记录epoch统计信息（均值、标准差、标准误差）
        epoch_mean, epoch_std, epoch_se = compute_epoch_statistics(epoch_accuracies)
        epoch_stds.append(epoch_std)
        
        # 记录epoch时间统计
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_episode_time = np.mean(episode_times)
        total_episode_time = np.sum(episode_times)
        
        # 获取当前本征支SupCon温度
        intrinsic_supcon_temp = model.get_intrinsic_supcon_stats().get('temperature', 0.0) if hasattr(model, 'get_intrinsic_supcon_stats') else 0.0
        
        print(f"  Epoch Summary: Loss={avg_epoch_loss:.4f}, Acc={avg_epoch_acc:.4f}, Std={epoch_std:.4f}")
        print(f"  SupCon: Intrinsic={avg_epoch_intrinsic_supcon:.4f}, Domain={avg_epoch_domain_supcon:.4f}, τ={intrinsic_supcon_temp:.3f}")
        print(f"  本征支相似度: Pos={avg_intrinsic_pos_sim:.3f}, Neg={avg_intrinsic_neg_sim:.3f}, CrossDom={avg_epoch_cross_sim:.3f}")
        print(f"  域支相似度: SDDC={avg_domain_cross_class_sim:.3f}, SDSC={avg_domain_same_class_sim:.3f}, CDSC={avg_domain_cdsc_sim:.3f}, CDDC={avg_domain_cddc_sim:.3f}")
        # ===== LeakIndex 日志输出 (新增) =====
        if not np.isnan(avg_leak):
            print(f"  LeakIndex: {avg_leak:.4f}±{std_leak:.4f}, Intensity={avg_intensity:.4f}")
        else:
            print(f"  LeakIndex: N/A (无有效数据)")
        print(f"  Time Summary: Epoch={epoch_time:.2f}s, Avg Episode={avg_episode_time:.2f}s")
        
        # 每 5 个 epoch 记录一次分离比和T-SNE可视化
        if (epoch + 1) % 5 == 0:
            # 若本 epoch 计算了分离比，则累加到曲线数据并打印均值
            if epoch_sep_ratios is not None and len(epoch_sep_ratios) > 0:
                avg_sep_ratio = float(np.mean(epoch_sep_ratios))
                sep_ratio_curve.append(avg_sep_ratio)
                sep_ratio_epochs.append(epoch + 1)
                print(f"  Separation Ratio (avg over epoch): {avg_sep_ratio:.4f}")
            
            # ===== T-SNE 可视化 (Refactored) =====
            tsne_metrics = tsne_visualizer.visualize(model, epoch + 1, tsne_save_dir)
            
            # 记录T-SNE指标
            tsne_metrics_history.append(tsne_metrics)
            tsne_epochs.append(epoch + 1)
            
            
        # 根据配置参数评估验证集性能
        if (epoch + 1) % Config.eval_frequency == 0:
            print("  Evaluating on validation set...")
            val_acc, val_lower, val_upper = evaluate_model(
                model, val_dataset, Config, num_test_episodes=Config.val_episodes
            )
            val_accuracies.append(val_acc)
            print(f"  Validation Accuracy: {val_acc:.4f} ({val_lower:.4f} ~ {val_upper:.4f})")
            
            # 收集热力图数据：[epoch, train_acc, val_acc, std]
            heatmap_data.append([epoch + 1, avg_epoch_acc, val_acc, epoch_std])
            
            # 保存最佳模型
            is_best_model = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best_model = True
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model_name': model.__class__.__name__,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'best_model.pth')
                print(f"  Saved new best model with validation accuracy: {best_val_acc:.4f}")
            


            # 每 10 个 epoch 绘制一次验证准确率曲线（若已收集到验证结果）
            if (epoch + 1) % 10 == 0:
                # 生成已评估的 epoch 列表
                current_val_epochs = [i * Config.eval_frequency for i in range(1, len(val_accuracies) + 1)]
                plot_val_accuracy_curve(
                    val_accuracies,
                    title=f"Validation Accuracy Curve (up to Epoch {epoch+1})",
                    val_epochs=current_val_epochs,
                    save_path=f"figures/val_accuracy_curve_epoch_{epoch+1}.png"
                )
                # 若已计算分离比，则同步绘制分离比变化曲线
                if len(sep_ratio_curve) > 0:
                    plot_separation_ratio_curve(
                        sep_ratios=sep_ratio_curve,
                        sep_epochs=sep_ratio_epochs,
                        title=f"Separation Ratio Curve (up to Epoch {epoch+1})",
                        save_path=f"figures/separation_ratio_curve_epoch_{epoch+1}.png"
                    )
    

    
    # 绘制改进的训练曲线 - 使用专业的准确率比较图
    # 根据实际验证次数计算val_epochs
    val_epochs = list(range(Config.eval_frequency, len(val_accuracies) * Config.eval_frequency + 1, Config.eval_frequency))
    
    # 绘制传统的训练曲线（包含损失）
    plot_training_curve(train_losses, train_accuracies, val_accuracies, 
                       title="Complete Training Curve", val_epochs=val_epochs,
                       save_path="figures/complete_training_curve.png")
    
    # 绘制训练与验证准确率对比图，直观展示模型在训练集与验证集上的性能差异
    plot_accuracy_comparison(train_accuracies, val_accuracies, 
                            title="Training vs Validation Accuracy Comparison",
                            val_epochs=val_epochs,
                            save_path="figures/accuracy_comparison.png")
    # 绘制训练阶段最终统计图，展示各 epoch 的平均准确率与标准差
    plot_epoch_statistics(train_accuracies, epoch_stds,
                          title="Training Statistics (Final)",
                          save_path="figures/epoch_stats_final.png")
    # 单独绘制训练阶段各 epoch 平均准确率曲线，便于观察整体趋势
    plot_epoch_accuracy(train_accuracies,
                        title="Epoch Average Accuracy (Final)",
                        save_path="figures/epoch_accuracy_final.png")
    
    # 绘制准确率热力图 - 行为不同epoch，列为指标
    if len(heatmap_data) > 0:
        # 转换为numpy数组：形状 [n_epochs, 3] -> (train_acc, val_acc, std)
        heatmap_matrix = np.array(heatmap_data)[:, 1:]  # 不转置，按 epoch 为行
        
        # 创建标签
        epoch_labels = [f"Epoch {int(data[0])}" for data in heatmap_data]
        metric_labels = ["训练准确率", "验证准确率", "标准差"]
        
        plot_accuracy_heatmap(
            heatmap_matrix,
            class_names=epoch_labels,
            metric_names=metric_labels,
            title="Training Progress Heatmap - Accuracy & Statistics",
            save_path="figures/training_progress_heatmap.png"
        )
        
        print(f"  Generated training progress heatmap with {len(heatmap_data)} validation points")
    
    # ===== T-SNE 聚类指标演化图 =====
    if len(tsne_metrics_history) > 0:
        plot_tsne_evolution(
            metrics_history=tsne_metrics_history,
            epochs=tsne_epochs,
            save_dir=tsne_save_dir,
            title="T-SNE 聚类指标演化 (ExpB1Model)"
        )
        print(f"  📈 T-SNE聚类指标演化图已生成，共 {len(tsne_epochs)} 个采样点")
    
    # ===== 对比学习相似度曲线可视化 =====
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(intrinsic_pos_sim_history) + 1)
    
    # 本征支相似度曲线
    ax1 = axes[0]
    ax1.plot(epochs_range, intrinsic_pos_sim_history, 'g-', label='同类正样本 (Pos)', linewidth=2)
    ax1.plot(epochs_range, intrinsic_neg_sim_history, 'r-', label='异类负样本 (Neg)', linewidth=2)
    ax1.plot(epochs_range, intrinsic_cross_domain_sim_history, 'b--', label='跨域同类 (CrossDom)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('余弦相似度')
    ax1.set_title('本征支 (Intrinsic) 相似度演化')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.0)
    
    # 域支相似度曲线
    ax2 = axes[1]
    ax2.plot(epochs_range, domain_cross_class_sim_history, 'g-', label='同域异类 (SDDC)', linewidth=2)
    ax2.plot(epochs_range, domain_same_class_sim_history, 'b-', label='同域同类 (SDSC)', linewidth=2)
    ax2.plot(epochs_range, domain_cdsc_sim_history, 'm--', label='跨域同类 (CDSC-Leak)', linewidth=2)
    ax2.plot(epochs_range, domain_cddc_sim_history, 'r:', label='跨域异类 (CDDC-Neg)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('余弦相似度')
    ax2.set_title('域支 (Domain) 相似度行为分析')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('figures/supcon_similarity_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 对比学习相似度曲线已保存到: figures/supcon_similarity_evolution.png")
    
    # ===== LeakIndex 可视化 (新增) =====
    if len(leakindex_history) > 0:
        # 过滤掉 NaN 值以便绘图
        valid_indices = [i for i, x in enumerate(leakindex_history) if not np.isnan(x)]
        if len(valid_indices) > 0:
            valid_epochs = [i + 1 for i in valid_indices]
            valid_leakindex = [leakindex_history[i] for i in valid_indices]
            valid_leakintensity = [leakintensity_history[i] for i in valid_indices]
            
            plot_leakindex_curve(
                leakindex_history=valid_leakindex,
                leakintensity_history=valid_leakintensity,
                epochs=valid_epochs,
                title="LeakIndex 演化曲线 (域支语义泄露监控)",
                save_path="figures/leakindex_evolution.png"
            )
            print(f"  📊 LeakIndex 曲线已保存到: figures/leakindex_evolution.png")
            print(f"     有效数据点: {len(valid_indices)}/{len(leakindex_history)} epochs")
        else:
            print(f"  ⚠️  LeakIndex 数据全部为 NaN，跳过可视化")
    else:
        print(f"  ⚠️  LeakIndex 历史记录为空，跳过可视化")
    
    # 计算并输出总训练时间统计
    total_training_time = time.time() - total_start_time
    avg_epoch_time = np.mean(epoch_times)
    total_episodes = Config.num_epochs * Config.episodes_per_epoch
    avg_episode_time_overall = total_training_time / total_episodes
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED - TIME STATISTICS")
    print("="*60)
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
    print(f"Number of Epochs: {Config.num_epochs}")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print(f"Total Episodes: {total_episodes}")
    print(f"Average Time per Episode: {avg_episode_time_overall:.2f}s")
    print(f"Episodes per Epoch: {Config.episodes_per_epoch}")
    print("="*60)
    print("All done!")


if __name__ == "__main__":
    main()
