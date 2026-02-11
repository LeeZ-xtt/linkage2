"""
简化版学习率调度器：线性预热（Linear Warmup）+ 多步下降（MultiStepLR）。

设计目标：
- 仅保留必要功能：在每个 epoch 开始时调用 step()，预热期间线性升温，结束后按里程碑衰减。
- 去除分布式同步、复杂校验和冗余日志，代码更精简、易读、易维护。

保留的外部接口：
- Scheduler.step() / get_lr() / get_group_lrs()
- Scheduler.state_dict() / load_state_dict()（仅用于基本保存/恢复，不做额外逻辑）

注意：
- 仍建议在“每个 epoch 开始”调用 step()，以便里程碑在指定 epoch 生效（例如 milestone=6 在进入第 6 个 epoch 的开头触发）。
"""

from typing import Any, Dict, List

import torch
from torch.optim.lr_scheduler import MultiStepLR
from config import Config


class Scheduler:
    """
    组合调度器：Linear Warmup + MultiStep Decay。

    - 在 warmup_epochs 期间使用 WarmupScheduler 线性提升学习率
    - 预热结束后自动切换到 MultiStepLR（主调度器保持原有方法和参数不变）
    - 每个 epoch 开始时调用 step() 更新学习率

    提供：
    - get_lr()：返回当前（第一个 param_group 的）学习率；如需全部分组，调用 get_group_lrs()
    - state_dict()/load_state_dict()：基础的保存/加载（为兼容现有代码而保留）
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_epochs: int | None = None,
        warmup_start_lr: float | None = None,
        milestones: List[int] | None = None,
        gamma: float | None = None,
        num_epochs: int | None = None,
        sync_distributed: bool = True,  # 为兼容旧接口，参数保留但不使用
        verbose: bool = False,  # 为兼容旧接口，参数保留但不使用
    ) -> None:
        # 基本参数与状态
        if len(optimizer.param_groups) == 0:
            raise ValueError("optimizer 必须至少包含一个 param_group")

        self.optimizer = optimizer

        # 先解析默认值，再进行校验，避免 None 比较错误
        w_epochs = int(Config.warmup_epochs if warmup_epochs is None else warmup_epochs)
        w_start_lr = float(Config.warmup_start_lr if warmup_start_lr is None else warmup_start_lr)
        milestones_list = list(Config.scheduler_milestones if milestones is None else milestones)
        gamma_val = float(Config.scheduler_gamma if gamma is None else gamma)
        num_epochs_val = int(Config.num_epochs if num_epochs is None else num_epochs)

        # 新增：从 Config 读取 warmup 开关，允许独立控制预热阶段
        warmup_enabled = bool(getattr(Config, "warmup", False))

        if w_epochs < 0:
            raise ValueError("warmup_epochs 必须是非负整数")

        # 当 warmup 未启用时，强制关闭预热阶段（与主衰减逻辑解耦）
        if not warmup_enabled:
            w_epochs = 0

        # 赋值到实例
        self.warmup_enabled = warmup_enabled
        self.warmup_epochs = w_epochs
        self.warmup_start_lr = w_start_lr
        self.milestones = milestones_list
        self.gamma = gamma_val
        self.num_epochs = num_epochs_val

        # 记录优化器初始的 base_lrs（用于预热目标）
        self.base_lrs: List[float] = [float(pg["lr"]) for pg in self.optimizer.param_groups]

        # 全局进度与阶段标记
        self.last_epoch: int = -1
        self.in_warmup: bool = (self.warmup_enabled and self.warmup_epochs > 0)

        # 主调度器（链式调用，不传显式 epoch）
        self._main = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)

    # -------------------- 调度接口 --------------------
    def step(self) -> None:
        """在每个 epoch 开始时调用，更新学习率。"""
        # 推进全局计数
        self.last_epoch += 1

        # 始终推进主调度器，以保证其里程碑与全局进度对齐
        self._main.step()

        # 预热阶段：用线性公式覆盖主调度器写入的 lr
        if self.in_warmup and self.last_epoch < self.warmup_epochs:
            progress = min((self.last_epoch + 1) / self.warmup_epochs, 1.0)
            warmup_lrs = [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
            for pg, lr in zip(self.optimizer.param_groups, warmup_lrs):
                pg["lr"] = float(lr)

        # 结束预热标记（从下一步开始完全按主调度器）
        if self.last_epoch >= self.warmup_epochs:
            self.in_warmup = False

    def get_lr(self) -> float:
        """
        返回当前学习率（第一个 param_group 的 lr）。
        若存在多 param_group，可使用 get_group_lrs() 获取全部。
        """
        return float(self.optimizer.param_groups[0]["lr"])

    def get_group_lrs(self) -> List[float]:
        """返回所有 param_group 的当前学习率。"""
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """返回简化的状态字典（为兼容现有代码）。"""
        return {
            "last_epoch": self.last_epoch,
            "in_warmup": self.in_warmup,
            "main_state": self._main.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """状态恢复：恢复基本进度与学习率。"""
        self.last_epoch = int(state_dict.get("last_epoch", -1))
        self.in_warmup = bool(state_dict.get("in_warmup", (self.warmup_enabled and self.last_epoch < self.warmup_epochs)))

        main_state = state_dict.get("main_state")
        if main_state:
            # 恢复主调度器状态
            self._main.load_state_dict(main_state)

        # 根据当前阶段设置优化器的学习率
        if self.in_warmup and self.last_epoch < self.warmup_epochs:
            progress = min((self.last_epoch + 1) / self.warmup_epochs, 1.0)
            lrs = [
                self.warmup_start_lr + progress * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            lrs = list(self._main.get_last_lr())
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = float(lr)

