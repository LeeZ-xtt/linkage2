import torch
import numpy as np
from utils.visualization import plot_tsne_embeddings
from config import Config

class TSNEVisualizer:
    """
    Manages t-SNE visualization with fixed episodes for consistency.
    """
    def __init__(self, dataset, n_way, k_shot, query_per_class, num_episodes, 
                 support_domain_pool, query_domain_pool, device):
        """
        Args:
            dataset: PACSDataset instance (should have eval_transform/no augmentation)
            n_way, k_shot, query_per_class: Episode parameters
            num_episodes: Number of episodes to fix and visualize
            support_domain_pool: List of allowed support domains
            query_domain_pool: List of allowed query domains
            device: torch.device
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.num_episodes = num_episodes
        self.support_domain_pool = support_domain_pool
        self.query_domain_pool = query_domain_pool
        self.device = device
        self.fixed_episodes = []
        # 预构建 (class_idx, domain_idx) -> [dataset_index, ...] 映射，避免每次 O(N) 扫描全数据集
        # dataset.labels: List[int] 真实类别ID; dataset.domain_labels: List[int] 域ID (0-3)
        self._indices_by_class_domain = self._build_indices_by_class_domain()
        self._init_fixed_episodes()
        
    def _build_indices_by_class_domain(self):
        """预构建索引映射: (class_idx, domain_idx) -> indices(list[int])"""
        mapping = {}
        # dataset.labels/domain_labels 为 python list，长度 = len(dataset)
        for idx, (l, d) in enumerate(zip(self.dataset.labels, self.dataset.domain_labels)):
            key = (int(l), int(d))
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(idx)
        return mapping

    def _init_fixed_episodes(self):
        """
        Pre-selects all indices and metadata for the episodes to ensure 
        they remain identical throughout training.
        """
        print(f"🔒 Initializing {self.num_episodes} fixed episodes for t-SNE visualization...")
        local_rng = np.random.RandomState(Config.seed + 42)
        
        for i in range(self.num_episodes):
            episode_data = {
                'support_indices': [],
                'support_labels': [],  # 全局真实类别ID
                'support_labels_episode': [],  # episode内部编号
                'support_domains': [],
                'query_indices': [],
                'query_labels': [],  # 全局真实类别ID
                'query_labels_episode': [],  # episode内部编号
                'query_domains': [],
            }
            
            query_domain = local_rng.choice(self.query_domain_pool)
            available_support = [d for d in self.support_domain_pool if d != query_domain] or self.support_domain_pool
            selected_class_indices = local_rng.choice(len(self.dataset.CATEGORIES), self.n_way, replace=False)
            
            for class_idx_in_episode, real_class_idx in enumerate(selected_class_indices):
                category = self.dataset.CATEGORIES[real_class_idx]
                
                # --- Support Set ---
                for _ in range(self.k_shot):
                    s_domain = local_rng.choice(available_support)
                    
                    # Find valid indices in dataset
                    valid_indices = self._get_indices_for(real_class_idx, s_domain)
                    if len(valid_indices) == 0:
                        raise ValueError(f"No samples found for class {category} in domain {s_domain}")
                        
                    idx = local_rng.choice(valid_indices)
                    episode_data['support_indices'].append(idx)
                    episode_data['support_labels'].append(real_class_idx)
                    episode_data['support_labels_episode'].append(class_idx_in_episode)
                    episode_data['support_domains'].append(self.dataset.DOMAINS.index(s_domain))
                    
                # --- Query Set ---
                valid_query_indices = self._get_indices_for(real_class_idx, query_domain)
                if len(valid_query_indices) == 0:
                    raise ValueError(f"No samples found for class {category} in query domain {query_domain}")
                if len(valid_query_indices) < self.query_per_class:
                    # Allow replacement if not enough samples
                    q_indices = local_rng.choice(valid_query_indices, self.query_per_class, replace=True)
                else:
                    q_indices = local_rng.choice(valid_query_indices, self.query_per_class, replace=False)
                
                episode_data['query_indices'].extend(q_indices)
                episode_data['query_labels'].extend([real_class_idx] * self.query_per_class)
                episode_data['query_labels_episode'].extend([class_idx_in_episode] * self.query_per_class)
                episode_data['query_domains'].extend([self.dataset.DOMAINS.index(query_domain)] * self.query_per_class)
            
            self.fixed_episodes.append(episode_data)
        
        print("✅ Fixed episodes initialized.")

    def _get_indices_for(self, class_idx, domain_name):
        """Find dataset indices for a specific class and domain."""
        domain_idx = self.dataset.DOMAINS.index(domain_name)
        return self._indices_by_class_domain.get((int(class_idx), int(domain_idx)), [])

    def visualize(self, model, epoch, save_dir):
        """
        Runs the visualization process using the fixed episodes.
        """
        model.set_mode('eval')
        
        tsne_features_list = []
        tsne_labels_list = []
        tsne_labels_global_list = []
        tsne_domain_labels_list = []
        all_support_features = []
        all_support_labels_global = []
        
        with torch.no_grad():
            for ep_data in self.fixed_episodes:
                s_imgs = self._load_images(ep_data['support_indices'])
                q_imgs = self._load_images(ep_data['query_indices'])
                
                s_lbls_episode = torch.tensor(ep_data['support_labels_episode']).to(self.device)
                q_lbls_episode = torch.tensor(ep_data['query_labels_episode']).to(self.device)
                s_lbls_global = torch.tensor(ep_data['support_labels']).to(self.device)
                q_lbls_global = torch.tensor(ep_data['query_labels']).to(self.device)
                s_doms = torch.tensor(ep_data['support_domains']).to(self.device)
                q_doms = torch.tensor(ep_data['query_domains']).to(self.device)
                
                s_feat, _ = model.extract_features(s_imgs)
                q_feat, _ = model.extract_features(q_imgs)
                
                all_support_features.append(s_feat.cpu())
                all_support_labels_global.append(s_lbls_global.cpu())
                
                all_feat = torch.cat([s_feat, q_feat], dim=0)  # [N_s + N_q, D]
                tsne_features_list.append(all_feat.cpu())
                tsne_labels_list.append(torch.cat([s_lbls_episode, q_lbls_episode], dim=0).cpu())
                tsne_labels_global_list.append(torch.cat([s_lbls_global, q_lbls_global], dim=0).cpu())
                tsne_domain_labels_list.append(torch.cat([s_doms, q_doms], dim=0).cpu())

        tsne_features = torch.cat(tsne_features_list, dim=0)            # [N_total, D]
        tsne_labels_episode = torch.cat(tsne_labels_list, dim=0)         # [N_total] (episode内编号，跨episode会冲突)
        tsne_labels_global = torch.cat(tsne_labels_global_list, dim=0)   # [N_total] (真实类别ID)
        tsne_domain_labels = torch.cat(tsne_domain_labels_list, dim=0)   # [N_total] (域ID)
        
        # 计算全局原型：按真实类别ID聚合所有episodes的support特征
        all_support_feat = torch.cat(all_support_features, dim=0)        # [N_support_total, D]
        all_support_lbls = torch.cat(all_support_labels_global, dim=0)   # [N_support_total]
        unique_real_classes = torch.unique(all_support_lbls).tolist()
        
        # 计算每个真实类别的全局原型
        global_prototypes_dict = {}
        for real_class_idx in unique_real_classes:
            mask = all_support_lbls == real_class_idx
            if mask.sum() > 0:
                global_prototypes_dict[real_class_idx] = all_support_feat[mask].mean(dim=0)  # [D]
        
        # 构建原型tensor和label映射
        proto_class_mapping = {}
        if global_prototypes_dict:
            proto_list = []
            for proto_idx, real_class_idx in enumerate(sorted(unique_real_classes)):
                proto_list.append(global_prototypes_dict[real_class_idx])
                proto_class_mapping[real_class_idx] = proto_idx
            tsne_prototypes = torch.stack(proto_list).cpu()  # [N_cls, D]
        else:
            tsne_prototypes = None
        
        # ===== 重要：可视化用 label 必须是全局一致的连续编号 =====
        # 旧逻辑使用 episode 内 label（0..n_way-1），多个 episode 拼接会导致不同真实类别被误染成同一类。
        if proto_class_mapping:
            labels_for_plot = torch.zeros_like(tsne_labels_global)
            for real_class_idx, proto_idx in proto_class_mapping.items():
                labels_for_plot[tsne_labels_global == real_class_idx] = proto_idx
            n_way_plot = len(proto_class_mapping)
            class_names_plot = [self.dataset.CATEGORIES[i] for i in sorted(unique_real_classes)]
        else:
            labels_for_plot = tsne_labels_episode
            n_way_plot = self.n_way
            class_names_plot = None
        
        # 计算高维指标
        if tsne_prototypes is not None and proto_class_mapping:
            high_dim_metrics = self._compute_high_dim_metrics(
                features=tsne_features,
                labels=labels_for_plot,
                prototypes=tsne_prototypes,
                n_way=n_way_plot
            )
        else:
            high_dim_metrics = {"silhouette_score": 0.0, "intra_class_dist": 0.0, 
                               "inter_class_dist": 0.0, "cluster_ratio": 0.0}

        print(f"  📊 Generating Fixed T-SNE (Epoch {epoch})...")
        metrics, _ = plot_tsne_embeddings(
            features=tsne_features,
            labels=labels_for_plot,
            epoch=epoch,
            n_way=n_way_plot,
            domain_labels=tsne_domain_labels,
            prototypes=tsne_prototypes,
            save_dir=save_dir,
            domain_names=self.dataset.DOMAINS,
            perplexity=20.0,
            random_state=Config.seed,
            class_names=class_names_plot,
            high_dim_metrics=high_dim_metrics
        )

        model.set_mode('train')
        return metrics

    def _compute_high_dim_metrics(self, features, labels, prototypes, n_way):
        import numpy as np
        from sklearn.metrics import silhouette_score
        import torch.nn.functional as F

        feat_cpu = features.detach().cpu() if isinstance(features, torch.Tensor) else torch.as_tensor(features)
        labels_cpu = labels.detach().cpu() if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        features_np = feat_cpu.numpy()
        labels_np = labels_cpu.numpy()

        # 计算轮廓系数
        if len(np.unique(labels_np)) > 1 and len(features_np) > 1:
            sample_size = min(300, len(features_np))
            try:
                kwargs = {'sample_size': sample_size, 'random_state': Config.seed} if sample_size < len(features_np) else {}
                silhouette = float(silhouette_score(features_np, labels_np, **kwargs))
            except Exception:
                silhouette = 0.0
        else:
            silhouette = 0.0

        # 计算类内/类间距离
        proto_cpu = prototypes.detach().cpu() if isinstance(prototypes, torch.Tensor) else (
            torch.as_tensor(prototypes) if prototypes is not None else None)
        
        if proto_cpu is None or proto_cpu.size(0) == 0:
            intra_class_dist = inter_class_dist = 0.0
        else:
            dists = torch.cdist(feat_cpu, proto_cpu, p=2)
            labels_onehot = F.one_hot(labels_cpu.to(torch.long), num_classes=n_way).bool()
            intra_dists = dists[labels_onehot]
            intra_class_dist = float(intra_dists.mean().item()) if intra_dists.numel() > 0 else 0.0
            inter_class_dist = float(torch.pdist(proto_cpu, p=2).mean().item()) if proto_cpu.size(0) > 1 else 0.0

        eps = 1e-8
        cluster_ratio = inter_class_dist / intra_class_dist if intra_class_dist > eps else (
            float('inf') if inter_class_dist > 0 else 0.0)

        return {
            "silhouette_score": silhouette,
            "intra_class_dist": intra_class_dist,
            "inter_class_dist": inter_class_dist,
            "cluster_ratio": cluster_ratio,
        }

    def _load_images(self, indices):
        """Batch load images from dataset by indices."""
        images = [self.dataset[idx][0] for idx in indices]
        return torch.stack(images).to(self.device)
