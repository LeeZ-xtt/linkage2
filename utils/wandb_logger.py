"""
WandB 日志记录工具类
提供简洁的接口来管理 WandB 日志记录功能
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from config import Config


class WandBLogger:
    """
    WandB 日志记录器，封装常用的日志记录功能
    """
    
    def __init__(self, enabled: bool = True):
        """
        初始化 WandB 日志记录器
        
        Args:
            enabled: 是否启用 WandB 日志记录
        """
        self.enabled = enabled and Config.use_wandb
        self.wandb = None
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, logging disabled")
                self.enabled = False
    
    def init(self, project_name: str = "deep-aggregation-few-shot", 
             experiment_name: Optional[str] = None,
             config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化 WandB 项目
        
        Args:
            project_name: 项目名称
            experiment_name: 实验名称
            config_dict: 配置字典
        """
        if not self.enabled:
            return
            
        # 默认配置
        default_config = {
            "learning_rate": Config.learning_rate,
            "momentum": Config.momentum,
            "weight_decay": Config.weight_decay,
            "num_epochs": Config.num_epochs,
            "n_way": Config.n_way,
            "k_shot": Config.k_shot,
            "query_per_class": Config.query_per_class,
            "seed": Config.seed,
            #"aggregation_layers": Config.aggregation_layers,
            "val_episodes": Config.val_episodes,
            "device": str(Config.device)
        }
        
        # 合并用户提供的配置
        if config_dict:
            default_config.update(config_dict)
        
        self.wandb.init(
            project=project_name,
            name=experiment_name,
            config=default_config
        )
        
        print(f"✅ WandB initialized: {project_name}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤数
        """
        if not self.enabled:
            return
            
        log_dict = metrics.copy()
        if step is not None:
            log_dict["step"] = step
            
        self.wandb.log(log_dict)
    
    def log_episode_metrics(self, loss: float, accuracy: float, 
                          episode: int, epoch: int) -> None:
        """
        记录 episode 级别的指标
        
        Args:
            loss: 损失值
            accuracy: 准确率
            episode: episode 编号
            epoch: epoch 编号
        """
        self.log_metrics({
            "episode_loss": loss,
            "episode_accuracy": accuracy,
            "episode": episode,
            "epoch": epoch
        })
    
    def log_epoch_metrics(self, loss: float, accuracy: float, 
                         learning_rate: float, epoch: int) -> None:
        """
        记录 epoch 级别的指标
        
        Args:
            loss: 平均损失
            accuracy: 平均准确率
            learning_rate: 学习率
            epoch: epoch 编号
        """
        self.log_metrics({
            "epoch_loss": loss,
            "epoch_accuracy": accuracy,
            "learning_rate": learning_rate,
            "epoch": epoch
        })
    
    def log_validation_metrics(self, accuracy: float, lower_bound: float, 
                             upper_bound: float, epoch: int) -> None:
        """
        记录验证集指标
        
        Args:
            accuracy: 验证准确率
            lower_bound: 置信区间下界
            upper_bound: 置信区间上界
            epoch: epoch 编号
        """
        self.log_metrics({
            "val_accuracy": accuracy,
            "val_accuracy_lower": lower_bound,
            "val_accuracy_upper": upper_bound,
            "val_confidence_interval": upper_bound - lower_bound,
            "epoch": epoch
        })
    
    def log_best_model(self, accuracy: float, epoch: int) -> None:
        """
        记录最佳模型信息
        
        Args:
            accuracy: 最佳验证准确率
            epoch: 对应的 epoch
        """
        self.log_metrics({
            "best_val_accuracy": accuracy,
            "best_model_epoch": epoch
        })
    
    def log_model_architecture(self, model: torch.nn.Module) -> None:
        """
        记录模型架构信息
        
        Args:
            model: PyTorch 模型
        """
        if not self.enabled:
            return
            
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log_metrics({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设 float32
        })
        
        # 记录模型结构
        if hasattr(self.wandb, 'watch'):
            self.wandb.watch(model, log="all", log_freq=100)
    
    def log_layer_weights(self, layer_weights: List[float], 
                         layer_names: List[str], epoch: int) -> None:
        """
        记录层权重信息
        
        Args:
            layer_weights: 层权重列表
            layer_names: 层名称列表
            epoch: epoch 编号
        """
        if not self.enabled:
            return
            
        # 记录每个层的权重
        weight_dict = {}
        for name, weight in zip(layer_names, layer_weights):
            weight_dict[f"layer_weight_{name}"] = weight
        
        # 添加权重统计信息
        weight_dict.update({
            "layer_weights_mean": np.mean(layer_weights.detach().cpu().numpy()),
            "layer_weights_std": np.std(layer_weights.detach().cpu().numpy()),
            "layer_weights_max": np.max(layer_weights.detach().cpu().numpy()),
            "layer_weights_min": np.min(layer_weights.detach().cpu().numpy()),
            "epoch": epoch
        })
        
        self.log_metrics(weight_dict)
    
    def log_gradient_norms(self, model: torch.nn.Module, epoch: int) -> None:
        """
        记录梯度范数信息
        
        Args:
            model: PyTorch 模型
            epoch: epoch 编号
        """
        if not self.enabled:
            return
            
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每层的梯度范数（可选，避免过多日志）
                if "weight" in name:  # 只记录权重层的梯度
                    self.log_metrics({
                        f"grad_norm_{name}": param_norm.item(),
                        "epoch": epoch
                    })
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.log_metrics({
                "grad_norm_total": total_norm,
                "epoch": epoch
            })
    
    def log_learning_curve_summary(self, train_losses: List[float], 
                                 train_accuracies: List[float],
                                 val_accuracies: List[float]) -> None:
        """
        记录训练曲线摘要信息
        
        Args:
            train_losses: 训练损失列表
            train_accuracies: 训练准确率列表
            val_accuracies: 验证准确率列表
        """
        if not self.enabled:
            return
            
        summary_metrics = {
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "final_train_accuracy": train_accuracies[-1] if train_accuracies else 0,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else 0,
            "best_val_accuracy": max(val_accuracies) if val_accuracies else 0,
            "train_loss_improvement": train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0,
            "train_accuracy_improvement": train_accuracies[-1] - train_accuracies[0] if len(train_accuracies) > 1 else 0
        }
        
        self.log_metrics(summary_metrics)
    
    def finish(self) -> None:
        """
        结束 WandB 会话
        """
        if self.enabled and self.wandb:
            self.wandb.finish()
            print("✅ WandB session finished")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.finish()


# 创建全局实例
logger = WandBLogger()