"""
数据加载器模块
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class PACSDataset(Dataset):
    """
    PACS 数据集 (Photo, Art Painting, Cartoon, Sketch)
    目录结构: root_dir/domain/category/image.jpg
    """
    DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
    CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __init__(self, root_dir, target_domains=None, split='train', transform=None):
        """
        初始化 PACS 数据集
        Args:
            root_dir (str): PACS 数据集根目录路径
            target_domains (list or str): 使用的域；None 表示全部域
            split (str): 'train' | 'val' | 'test' | 'all'
            transform (callable): 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"数据集根目录不存在: {root_dir}")

        if target_domains is None:
            self.target_domains = self.DOMAINS
        elif isinstance(target_domains, str):
            self.target_domains = [target_domains]
        else:
            self.target_domains = target_domains

        for d in self.target_domains:
            if d not in self.DOMAINS:
                raise ValueError(f"无效的域: {d}. 必须是 {self.DOMAINS} 之一")

        self.image_paths = []
        self.labels = []        # 类别标签 (0-6)
        self.domain_labels = [] # 域标签 (0-3)

        self._load_data()

        # 创建 类别->索引 映射，用于 episode 采样
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def _load_data(self):
        """加载数据路径和标签"""
        for domain_idx, domain in enumerate(self.DOMAINS):
            if domain not in self.target_domains:
                continue

            domain_dir = os.path.join(self.root_dir, domain)
            if not os.path.exists(domain_dir):
                print(f"警告: 域目录不存在: {domain_dir}")
                continue

            for class_idx, category in enumerate(self.CATEGORIES):
                category_dir = os.path.join(domain_dir, category)
                if not os.path.exists(category_dir):
                    continue

                for img_name in os.listdir(category_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(category_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
                        self.domain_labels.append(domain_idx)

        if len(self.image_paths) == 0:
            print(f"警告: 未找到任何图像。请检查路径: {self.root_dir} 和目标域: {self.target_domains}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        Returns:
            image: Tensor [C, H, W] - 图像张量 (C=3通道, H=高度, W=宽度)
            label: int - 类别标签 (0-6)
            domain_label: int - 域标签 (0-3)
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.image_paths)})")

        img_path = self.image_paths[idx]
        label = self.labels[idx]
        domain_label = self.domain_labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')  # PIL Image: (H, W, 3)
            if self.transform:
                image = self.transform(image)  # Tensor: [C, H, W] 经过transform后的张量
            # 返回: (image[C,H,W], label, domain_label)
            return image, label, domain_label
        except Exception as e:
            raise IOError(f"加载图像出错 {img_path}: {str(e)}")


def get_pacs_transform(image_size=224, split='train'):
    """
    获取 PACS 数据集的标准变换
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])



def create_cross_domain_episode_loader(dataset, n_way, k_shot, query_per_class, num_episodes, 
                                     support_domain_pool=None, query_domain_pool=None):
    """
    创建跨域episode加载器
    
    Args:
        dataset: PACSDataset实例
        n_way: 类别数
        k_shot: 支持样本数
        query_per_class: 查询样本数
        num_episodes: episode数量
        support_domain_pool: 支持集可选域列表
        query_domain_pool: 查询集可选域列表
        
    Returns:
        generator: 生成episode数据
    """
    if support_domain_pool is None:
        support_domain_pool = dataset.target_domains
    if query_domain_pool is None:
        query_domain_pool = dataset.target_domains
        
    return _create_cross_domain_episodes(dataset, n_way, k_shot, query_per_class, 
                                       num_episodes, support_domain_pool, query_domain_pool)

def _create_cross_domain_episodes(dataset, n_way, k_shot, query_per_class, num_episodes, 
                                support_domain_pool, query_domain_pool):
    """
    生成跨域episodes
    策略：
    1. 每个episode随机选择一个查询域 (from query_domain_pool)
    2. 支持集从 support_domain_pool 中采样，但必须排除当前选定的查询域
       (确保支持集和查询集来自完全不同的域，实现严格的跨域/域泛化)
    """
    for _ in range(num_episodes):
        # 1. 确定查询域
        query_domain = np.random.choice(query_domain_pool)  # 从query_domain_pool中随机选择一个域
        
        # 2. 确定支持域池 (修改：排除已选的查询域)
        # 过滤掉与当前查询域相同的支持域，确保严格的跨域设置
        available_support_domains = [d for d in support_domain_pool if d != query_domain]
        
        if not available_support_domains:
            # 如果过滤后为空，说明 support_pool 仅包含 query_domain，无法构成跨域
            raise ValueError(f"无法构建跨域Episode: 在排除查询域 '{query_domain}' 后，支持域池为空。\n"
                             f"当前支持域池: {support_domain_pool}\n"
                             f"请确保支持域池中包含除查询域以外的其他域。")
        
        # 随机选择n_way个类别
        selected_labels = np.random.choice(len(dataset.CATEGORIES), n_way, replace=False)  # 从CATEGORIES中随机选择n_way个类别
        
        # 初始化列表用于收集样本
        support_images, support_labels = [], []  # 支持集图像列表和标签列表
        query_images, query_labels = [], []      # 查询集图像列表和标签列表
        support_domains, query_domains = [], []  # 域标签列表
        
        for class_idx, category_idx in enumerate(selected_labels):
            category = dataset.CATEGORIES[category_idx]
            
            # --- 采样支持集 ---
            # 为该类别的每个shot随机选择一个域 (从 过滤后的 available_support_domains)
            for _ in range(k_shot):
                s_domain = np.random.choice(available_support_domains)  # 从不包含查询域的池中选择
                # 采样一张图片
                s_paths = _get_category_domain_samples(dataset, category, s_domain, 1)
                path = s_paths[0]
                
                # img: Tensor [C, H, W] - 单张图像张量
                img = dataset.transform(Image.open(path).convert('RGB')) if dataset.transform else Image.open(path).convert('RGB')
                support_images.append(img)  # 添加到列表，最终形状: List[Tensor[C,H,W]], len=n_way*k_shot
                support_labels.append(class_idx)  # 添加类别索引 (0 到 n_way-1)
                support_domains.append(dataset.DOMAINS.index(s_domain))  # 添加域索引 (0-3)
            
            # --- 采样查询集 ---
            # 全部来自 query_domain
            q_paths = _get_category_domain_samples(dataset, category, query_domain, query_per_class)  # 从query_domain中采样query_per_class张图片
            for path in q_paths:
                # img: Tensor [C, H, W] - 单张图像张量
                img = dataset.transform(Image.open(path).convert('RGB')) if dataset.transform else Image.open(path).convert('RGB')
                query_images.append(img)  # 添加到列表，最终形状: List[Tensor[C,H,W]], len=n_way*query_per_class
                query_labels.append(class_idx)  # 添加类别索引 (0 到 n_way-1)
                query_domains.append(dataset.DOMAINS.index(query_domain))  # 添加域索引 (0-3)
                
        # 返回一个episode的所有数据，张量维度说明:
        # support_images: [n_way*k_shot, C, H, W] - 支持集图像批次
        # support_labels: [n_way*k_shot] - 支持集标签 (值范围: 0 到 n_way-1)
        # query_images: [n_way*query_per_class, C, H, W] - 查询集图像批次
        # query_labels: [n_way*query_per_class] - 查询集标签 (值范围: 0 到 n_way-1)
        # support_domains: [n_way*k_shot] - 支持集域标签 (值范围: 0-3, 对应DOMAINS索引)
        # query_domains: [n_way*query_per_class] - 查询集域标签 (值范围: 0-3, 全部相同值)
        yield (torch.stack(support_images),      # [n_way*k_shot, C, H, W]
               torch.tensor(support_labels),     # [n_way*k_shot]
               torch.stack(query_images),        # [n_way*query_per_class, C, H, W]
               torch.tensor(query_labels),       # [n_way*query_per_class]
               torch.tensor(support_domains),    # [n_way*k_shot]
               torch.tensor(query_domains))      # [n_way*query_per_class]

def _get_category_domain_samples(dataset, category, domain, num_samples):
    """从指定类别和域中获取样本路径"""
    domain_idx = dataset.DOMAINS.index(domain)
    category_idx = dataset.CATEGORIES.index(category)
    
    # 找到同时满足类别和域条件的样本
    valid_indices = []
    for idx, (label, domain_label) in enumerate(zip(dataset.labels, dataset.domain_labels)):
        if label == category_idx and domain_label == domain_idx:
            valid_indices.append(idx)
    
    # 样本不足时使用替换采样，否则不放回采样
    if len(valid_indices) < num_samples:
        selected_indices = np.random.choice(valid_indices, num_samples, replace=True)
    else:
        selected_indices = np.random.choice(valid_indices, num_samples, replace=False)
    
    return [dataset.image_paths[idx] for idx in selected_indices]



if __name__ == "__main__":
    # 快速功能测试
    pacs_root = r"E:\PACS"
    if os.path.exists(pacs_root):
        try:
            # 测试多域采样
            ds = PACSDataset(pacs_root, target_domains=['photo', 'art_painting'], 
                           split='train', transform=get_pacs_transform(84, 'train'))
            loader = create_episode_loader(ds, n_way=5, k_shot=1, query_per_class=3, num_episodes=2)
            batch = next(iter(loader))
            print(f"✅ PACS数据加载器测试通过: support={batch[0].shape}, query={batch[2].shape}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  PACS数据集不存在，跳过测试")
