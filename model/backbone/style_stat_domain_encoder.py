"""
Style/Stat Domain Encoder ( 固定/弱可学习滤波器组 + 显式统计编码)

目标:
- 输出 f_dom: [B, domain_dim]，更像连续的“风格/工况坐标”，而不是域ID分类器特征。
- 主要由低阶统计/纹理能量驱动：边缘密度、方向纹理、频带能量、对比度等。
- 禁用 BatchNorm 依赖（episodic/多域混合稳定）；优先 GroupNorm/LayerNorm。
- AMP 下可稳定工作；输出尺度可控；梯度仅流向“弱可学习参数 + token head”。
"""


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StyleStatDomainEncoderConfig:
    """风格统计域编码器配置（与 config.py 对齐的最小集合）"""

    domain_dim: int = 64    # 输出维度（连续工况坐标）
    kernel_size: int = 7    # 固定滤波器组大小
    use_gabor: bool = True    # 是否使用 Gabor 滤波器
    use_dog: bool = True    # 是否使用 DoG 滤波器
    trainable_gain: bool = True    # 是否可学习 gain
    trainable_delta_kernel: bool = False    # 是否可学习 delta 核
    delta_kernel_eps: float = 1e-3    # delta 核的 eps
    energy_eps: float = 1e-6    # 能量 eps
    use_gram_stats: bool = True    # 是否使用 Gram 统计
    gram_rank: int = 16    # Gram 统计的 rank
    gram_out_dim: int = 64    # Gram 统计的输出维度
    token_dropout: float = 0.4    # token 的 dropout
    token_scale_init: float = 1.0    # token 的 scale 初始化
    token_hidden_dim: int = 256    # token 的隐藏维度
    norm: str = "gn"    # 归一化设置：明确禁用 BN


def _as_tensor(x: torch.Tensor) -> torch.Tensor:
    # 保持 torch.compile 友好：这里不引入动态 Python 逻辑
    return x


def _make_meshgrid(kernel_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 [-r, r] 的二维坐标网格，返回 X, Y，形状均为 [K, K]"""
    r = kernel_size // 2
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)  # [K]
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # [K, K], [K, K]
    return xx, yy


def _normalize_kernel(k: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按 L1 范数归一化核，保证不同核的响应尺度可比；对零均值核也适用。"""
    denom = k.abs().sum().clamp_min(eps)
    return k / denom


def _embed_center_kernel(small: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """将小核（如 3x3）嵌入到 KxK 的中心，返回 [K, K]。"""
    k = kernel_size
    out = torch.zeros((k, k), dtype=small.dtype, device=small.device)  # [K, K]
    s = small.shape[-1]
    r_big = k // 2
    r_small = s // 2
    out[(r_big - r_small) : (r_big + r_small + 1), (r_big - r_small) : (r_big + r_small + 1)] = small
    return out


def _scharr_kernels_3x3(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scharr X/Y，形状 [3,3]"""
    gx = torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        device=device,
        dtype=torch.float32,
    )
    gy = gx.t().contiguous()
    return gx, gy


def _laplacian_kernel_3x3(device: torch.device, mode: str = "8") -> torch.Tensor:
    """Laplacian 3x3：mode="4" or "8"。"""
    if mode == "4":
        k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    else:
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], device=device, dtype=torch.float32)
    return k


def _gaussian2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """2D Gaussian kernel，形状 [K, K]，sum=1。"""
    xx, yy = _make_meshgrid(kernel_size, device=device)  # [K, K], [K, K]
    s2 = float(sigma) * float(sigma)
    g = torch.exp(-(xx**2 + yy**2) / (2.0 * s2))  # [K, K]
    g = g / g.sum().clamp_min(1e-8)  # [K, K]
    return g


def _dog_kernel(kernel_size: int, sigma1: float, sigma2: float, device: torch.device) -> torch.Tensor:
    """Difference of Gaussians (DoG) kernel，形状 [K, K]，零均值。"""
    g1 = _gaussian2d(kernel_size, sigma1, device=device)  # [K, K]
    g2 = _gaussian2d(kernel_size, sigma2, device=device)  # [K, K]
    dog = g1 - g2  # [K, K]
    dog = dog - dog.mean()  # [K, K]
    dog = _normalize_kernel(dog)  # [K, K]
    return dog


def _gabor_kernel(
    kernel_size: int,
    theta_deg: float,
    lambd: float,
    sigma: float,
    gamma: float,
    psi: float,
    device: torch.device,
) -> torch.Tensor:
    """Gabor kernel，形状 [K, K]，零均值并归一化。"""
    theta = torch.tensor(theta_deg * torch.pi / 180.0, device=device, dtype=torch.float32)  # []
    xx, yy = _make_meshgrid(kernel_size, device=device)  # [K, K], [K, K]

    # 旋转坐标
    x_theta = xx * torch.cos(theta) + yy * torch.sin(theta)  # [K, K]
    y_theta = -xx * torch.sin(theta) + yy * torch.cos(theta)  # [K, K]

    # Gabor: exp(- (x'^2 + gamma^2 y'^2)/(2 sigma^2)) * cos(2π x'/λ + ψ)
    s2 = float(sigma) * float(sigma)
    env = torch.exp(-(x_theta**2 + (gamma**2) * (y_theta**2)) / (2.0 * s2))  # [K, K]
    carrier = torch.cos((2.0 * torch.pi * x_theta / float(lambd)) + float(psi))  # [K, K]
    g = env * carrier  # [K, K]

    # 去 DC，降低“亮度/语义”泄漏
    g = g - g.mean()  # [K, K]
    g = _normalize_kernel(g)  # [K, K]
    return g


class FixedFilterBank2D(nn.Module):
    """
    固定/弱可学习滤波器组。

    输入: x [B, 3, H, W] 或 [B, 1, H, W]
    输出: r [B, C_f, H, W]
    """

    def __init__(
        self,
        kernel_size: int = 7,
        use_gabor: bool = True,
        use_dog: bool = True,
        trainable_gain: bool = True,
        trainable_delta_kernel: bool = False,
        delta_kernel_eps: float = 1e-3,
        laplacian_mode: str = "8",
    ) -> None:
        super().__init__()
        assert kernel_size in (7, 9), "建议 kernel_size=7 或 9，便于固定滤波器堆叠"
        assert laplacian_mode in ("4", "8")
        self.kernel_size = int(kernel_size)
        self.use_gabor = bool(use_gabor)
        self.use_dog = bool(use_dog)
        self.trainable_gain = bool(trainable_gain)
        self.trainable_delta_kernel = bool(trainable_delta_kernel)
        self.delta_kernel_eps = float(delta_kernel_eps)
        self.laplacian_mode = str(laplacian_mode)

        # 灰度化系数（buffer，确保 device/dtype 自动迁移，且无梯度）
        rgb2gray = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)  # [3]
        self.register_buffer("rgb2gray", rgb2gray, persistent=True)

        # 构建固定核（以 CPU float32 创建，注册为 buffer；forward 时自动到同 device）
        weight = self._build_filter_bank(kernel_size=self.kernel_size)  # [C_f, 1, K, K]
        self.register_buffer("base_weight", weight, persistent=True)

        n_filters = int(weight.shape[0])

        # 弱可学习 gain：每个滤波器一个缩放
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(n_filters, dtype=torch.float32))  # [C_f]
        else:
            self.register_buffer("gain", torch.ones(n_filters, dtype=torch.float32), persistent=True)  # [C_f]

        # 可选：残差微调核 ΔW（默认关闭）
        if self.trainable_delta_kernel:
            # 仅学习一个“方向”，并在 forward 中做 per-filter Frobenius norm 归一化，再乘 eps
            self.delta_weight = nn.Parameter(torch.zeros_like(self.base_weight, dtype=torch.float32))  # [C_f,1,K,K]
        else:
            self.delta_weight = None

    @property
    def n_filters(self) -> int:
        return int(self.base_weight.shape[0])

    def _build_filter_bank(self, kernel_size: int) -> torch.Tensor:
        """
        返回固定核权重: [C_f, 1, K, K] (float32, CPU)
        """
        device = torch.device("cpu")

        kernels: list[torch.Tensor] = []

        # 一阶梯度：Scharr X/Y（嵌入到 KxK）
        gx3, gy3 = _scharr_kernels_3x3(device=device)  # [3,3], [3,3]
        gx = _embed_center_kernel(_normalize_kernel(gx3), kernel_size=kernel_size)  # [K,K]
        gy = _embed_center_kernel(_normalize_kernel(gy3), kernel_size=kernel_size)  # [K,K]
        kernels.append(gx)
        kernels.append(gy)

        # 二阶：Laplacian（嵌入到 KxK）
        lap3 = _laplacian_kernel_3x3(device=device, mode=self.laplacian_mode)  # [3,3]
        lap = _embed_center_kernel(_normalize_kernel(lap3), kernel_size=kernel_size)  # [K,K]
        kernels.append(lap)

        # 多尺度 DoG
        if self.use_dog:
            # 至少 3 组：使用 (0.6,1.0), (1.0,1.6), (0.6,1.6)
            for s1, s2 in [(0.6, 1.0), (1.0, 1.6), (0.6, 1.6)]:
                kernels.append(_dog_kernel(kernel_size, s1, s2, device=device))  # [K,K]

        # Gabor bank：4 方向 × 2 尺度
        if self.use_gabor:
            # λ 随 kernel_size 变化给出合理默认（避免“学回语义”的过细纹理）
            lambd_small = float(max(2, kernel_size // 3))
            lambd_mid = float(max(3, kernel_size // 2))
            # σ 经验设置：与 λ 成比例，保证核覆盖多个周期但不过度尖锐
            sigma_small = 0.56 * lambd_small
            sigma_mid = 0.56 * lambd_mid
            gamma = 0.5
            psi = 0.0
            for theta in (0.0, 45.0, 90.0, 135.0):
                kernels.append(
                    _gabor_kernel(kernel_size, theta_deg=theta, lambd=lambd_small, sigma=sigma_small, gamma=gamma, psi=psi, device=device)
                )  # [K,K]
                kernels.append(
                    _gabor_kernel(kernel_size, theta_deg=theta, lambd=lambd_mid, sigma=sigma_mid, gamma=gamma, psi=psi, device=device)
                )  # [K,K]

        weight = torch.stack(kernels, dim=0)  # [C_f, K, K]
        weight = weight.unsqueeze(1).contiguous()  # [C_f, 1, K, K]
        return weight.to(dtype=torch.float32, device=torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] or [B, 1, H, W]
        Returns:
            r: [B, C_f, H, W]
        """
        x = _as_tensor(x)
        b, c, h, w = x.shape  # [B,C,H,W]

        # 灰度化（减少语义，强化结构/纹理）
        if c == 3:
            coeff = self.rgb2gray.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)  # [1,3,1,1]
            x_gray = (x * coeff).sum(dim=1, keepdim=True)  # [B,1,H,W]
        elif c == 1:
            x_gray = x  # [B,1,H,W]
        else:
            raise ValueError(f"FixedFilterBank2D expects C=1 or C=3, got C={c}")

        # 组装权重：固定 base_weight + 可选极小 ΔW
        base_w = self.base_weight.to(device=x.device, dtype=torch.float32)  # [C_f,1,K,K] (fp32)
        if self.delta_weight is not None:
            delta = self.delta_weight.to(device=x.device, dtype=torch.float32)  # [C_f,1,K,K] (fp32)
            # per-filter Frobenius norm 归一化，控制自由度（弱可学习）
            delta_norm = delta.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)  # [C_f,1]
            delta_unit = (delta / delta_norm.view(-1, 1, 1, 1))  # [C_f,1,K,K]
            w = base_w + (self.delta_kernel_eps * delta_unit)  # [C_f,1,K,K]
        else:
            w = base_w  # [C_f,1,K,K]

        # 计算滤波响应（conv 在 fp32 更稳，之后再 cast 回 x.dtype）
        x32 = x_gray.to(dtype=torch.float32)  # [B,1,H,W]
        r32 = F.conv2d(x32, w, bias=None, stride=1, padding=self.kernel_size // 2)  # [B,C_f,H,W]

        # 逐滤波器 gain（弱可学习）
        g = self.gain.to(device=x.device, dtype=torch.float32).view(1, -1, 1, 1)  # [1,C_f,1,1]
        r32 = r32 * g  # [B,C_f,H,W]

        r = r32.to(dtype=x.dtype)  # [B,C_f,H,W]
        return r


class StyleEnergyNonlinearity(nn.Module):
    """
    将滤波响应 r 转为能量图 e，并做无 BN 的归一化。

    输入: r [B, C, H, W]
    输出: e [B, C, H, W]
    """

    def __init__(self, n_channels: int, energy_eps: float = 1e-6, norm: str = "gn") -> None:
        super().__init__()
        self.energy_eps = float(energy_eps)
        assert norm in ("gn", "ln")
        if norm == "gn":
            # num_groups=1 相当于对通道做 LN 风格归一化，但不依赖 batch 统计
            self.norm = nn.GroupNorm(num_groups=1, num_channels=n_channels, eps=1e-5, affine=True)
        else:
            # LayerNorm 需要 [B,C,H,W] -> LN over C；用 GroupNorm(1, C) 更自然
            self.norm = nn.GroupNorm(num_groups=1, num_channels=n_channels, eps=1e-5, affine=True)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = _as_tensor(r)
        # 能量映射：sqrt(r^2 + eps)，AMP 下用 fp32 计算更稳
        r32 = r.to(dtype=torch.float32)  # [B,C,H,W]
        e32 = torch.sqrt(r32 * r32 + self.energy_eps)  # [B,C,H,W]
        e = e32.to(dtype=r.dtype)  # [B,C,H,W]
        e = self.norm(e)  # [B,C,H,W]
        return e


class MultiScaleStatPool(nn.Module):
    """
    多尺度显式统计池化：
    - 每尺度提取 per-channel mean 与 log-std
    - 可选低秩 Gram 统计（Cr=16, d_gram=64 默认）

    输入: e [B, C, H, W]
    输出: s [B, D_stat]
    """

    def __init__(
        self,
        in_channels: int,
        use_gram_stats: bool = True,
        gram_rank: int = 16,
        gram_out_dim: int = 64,
        energy_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_gram_stats = bool(use_gram_stats)
        self.gram_rank = int(gram_rank)
        self.gram_out_dim = int(gram_out_dim)
        self.energy_eps = float(energy_eps)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)

        if self.use_gram_stats:
            # 方案A风格：固定随机投影到 Cr，再做弱可学习 gain（尽量难携带语义）
            # 1) 固定投影矩阵 P: [Cr, C]
            # 重要：不要污染全局 RNG（否则会影响训练采样/增强的随机性）
            gen = torch.Generator(device="cpu").manual_seed(0)
            proj = torch.randn(self.gram_rank, self.in_channels, dtype=torch.float32, generator=gen)  # [Cr, C]
            proj = proj / proj.norm(dim=1, keepdim=True).clamp_min(1e-8)  # [Cr, C]
            self.register_buffer("gram_proj_in", proj, persistent=True)

            # 2) 弱可学习 gain：对 Cr 维度逐通道缩放
            self.gram_gain = nn.Parameter(torch.ones(self.gram_rank, dtype=torch.float32))  # [Cr]

            # 3) 固定投影压缩到 d_gram（从上三角向量维度 T=Cr(Cr+1)/2）
            tri_dim = self.gram_rank * (self.gram_rank + 1) // 2
            proj_out = torch.randn(self.gram_out_dim, tri_dim, dtype=torch.float32, generator=gen)  # [d_gram, T]
            proj_out = proj_out / proj_out.norm(dim=1, keepdim=True).clamp_min(1e-8)  # [d_gram, T]
            self.register_buffer("gram_proj_out", proj_out, persistent=True)
        else:
            self.gram_proj_in = None
            self.gram_gain = None
            self.gram_proj_out = None

    def _stats(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # e: [B,C,H,W]
        mu = e.mean(dim=(2, 3))  # [B,C]
        var = e.var(dim=(2, 3), unbiased=False)  # [B,C]
        log_std = torch.log(torch.sqrt(var + self.energy_eps))  # [B,C]
        return mu, log_std

    def _low_rank_gram(self, e: torch.Tensor) -> torch.Tensor:
        """
        低秩 Gram:
        - 仅在 scale0 上计算，降低算力与容量
        - e: [B, C, H, W]
        输出: g: [B, d_gram]
        """
        b, c, h, w = e.shape  # [B,C,H,W]
        hw = float(h * w)

        # 固定投影到 Cr：E_r = P @ E，其中 E 展平成 [B, C, HW]
        e32 = e.to(dtype=torch.float32)  # [B,C,H,W]
        E = e32.flatten(2)  # [B,C,HW]
        P = self.gram_proj_in.to(device=e.device, dtype=torch.float32)  # [Cr,C]
        Er = torch.einsum("rc,bch->brh", P, E)  # [B,Cr,HW]

        # 弱可学习 gain
        gg = self.gram_gain.to(device=e.device, dtype=torch.float32).view(1, -1, 1)  # [1,Cr,1]
        Er = Er * gg  # [B,Cr,HW]

        # Gram: [B,Cr,Cr]
        G = torch.matmul(Er, Er.transpose(1, 2)) / max(hw, 1.0)  # [B,Cr,Cr]

        # 取上三角向量化: [B,T]
        idx = torch.triu_indices(self.gram_rank, self.gram_rank, device=G.device)  # [2,T]
        tri = G[:, idx[0], idx[1]]  # [B,T]

        # 固定投影到 d_gram
        W = self.gram_proj_out.to(device=e.device, dtype=torch.float32)  # [d_gram,T]
        g = F.linear(tri, W, bias=None)  # [B,d_gram]
        return g.to(dtype=e.dtype)  # [B,d_gram]

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e = _as_tensor(e)
        # scale0
        mu0, ls0 = self._stats(e)  # [B,C], [B,C]
        # scale1
        e1 = self.pool2(e)  # [B,C,H/2,W/2]
        mu1, ls1 = self._stats(e1)  # [B,C], [B,C]
        # scale2
        e2 = self.pool4(e)  # [B,C,H/4,W/4]
        mu2, ls2 = self._stats(e2)  # [B,C], [B,C]

        parts = [mu0, ls0, mu1, ls1, mu2, ls2]  # 6 * [B,C]
        if self.use_gram_stats:
            g = self._low_rank_gram(e)  # [B,d_gram]
            parts.append(g)

        s = torch.cat(parts, dim=1)  # [B,D_stat]
        return s


class DomainTokenHead(nn.Module):
    """
    统计向量 -> 域 token（连续工况坐标）。

    输入: s [B, D_stat]
    输出: f_dom [B, domain_dim]（tanh + 可学习 scale）
    """

    def __init__(
        self,
        in_dim: int,
        domain_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        scale_init: float = 1.0,
        output_logvar: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.domain_dim = int(domain_dim)
        self.output_logvar = bool(output_logvar)

        self.ln = nn.LayerNorm(self.in_dim, eps=1e-5)
        self.fc1 = nn.Linear(self.in_dim, int(hidden_dim), bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), self.domain_dim, bias=True)

        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

        if self.output_logvar:
            self.fc_logvar = nn.Linear(int(hidden_dim), self.domain_dim, bias=True)
        else:
            self.fc_logvar = None

        self._init_weights()

    def _init_weights(self) -> None:
        # 截断正态初始化（与项目里 projection head 风格一致）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s = _as_tensor(s)
        s = self.ln(s)  # [B,D_stat]
        h = self.fc1(s)  # [B,hidden]
        h = self.act(h)  # [B,hidden]
        h = self.drop(h)  # [B,hidden]

        t = self.fc2(h)  # [B,domain_dim]
        t = torch.tanh(t)  # [B,domain_dim]

        # scale 约束：避免幅值漂移导致 SupCon 不稳定
        scale = self.scale.clamp(0.1, 10.0)  # []
        f_dom = t * scale  # [B,domain_dim]

        if self.fc_logvar is None:
            return f_dom, None

        logvar = self.fc_logvar(h)  # [B,domain_dim]
        logvar = logvar.clamp(-6.0, 2.0)  # [B,domain_dim]
        return f_dom, logvar


class StyleStatDomainEncoder(nn.Module):
    """
    总封装：FixedFilterBank2D -> Energy -> MultiScaleStatPool -> TokenHead

    默认 forward(x) 返回 f_dom [B, domain_dim]（与旧 DomainEncoder 调用方式对齐，仅维度变更）。
    """

    def __init__(self, cfg: Optional[StyleStatDomainEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = StyleStatDomainEncoderConfig(**kwargs)
        self.cfg = cfg

        self.filter_bank = FixedFilterBank2D(
            kernel_size=cfg.kernel_size,
            use_gabor=cfg.use_gabor,
            use_dog=cfg.use_dog,
            trainable_gain=cfg.trainable_gain,
            trainable_delta_kernel=cfg.trainable_delta_kernel,
            delta_kernel_eps=cfg.delta_kernel_eps,
        )

        self.energy = StyleEnergyNonlinearity(
            n_channels=self.filter_bank.n_filters,
            energy_eps=cfg.energy_eps,
            norm=cfg.norm,
        )

        self.stat_pool = MultiScaleStatPool(
            in_channels=self.filter_bank.n_filters,
            use_gram_stats=cfg.use_gram_stats,
            gram_rank=cfg.gram_rank,
            gram_out_dim=cfg.gram_out_dim,
            energy_eps=cfg.energy_eps,
        )

        # D_stat = 6*C + (d_gram if enabled)
        c = self.filter_bank.n_filters
        d_stat = 6 * c + (cfg.gram_out_dim if cfg.use_gram_stats else 0)

        self.token_head = DomainTokenHead(
            in_dim=d_stat,
            domain_dim=cfg.domain_dim,
            hidden_dim=cfg.token_hidden_dim,
            dropout=cfg.token_dropout,
            scale_init=cfg.token_scale_init,
            output_logvar=False,
        )

    @property
    def feature_dim(self) -> int:
        return int(self.cfg.domain_dim)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        Args:
            x: [B,3,H,W]
            return_stats: 是否返回诊断信息（默认 False）
        Returns:
            f_dom: [B,domain_dim] 或 (f_dom, stats)
        """
        # Fixed filters
        r = self.filter_bank(x)  # [B,Cf,H,W]
        # Energy + norm (no BN)
        e = self.energy(r)  # [B,Cf,H,W]
        # Multi-scale explicit stats (+ low-rank gram)
        s = self.stat_pool(e)  # [B,D_stat]
        # Token head with scale control
        f_dom, _ = self.token_head(s)  # [B,domain_dim]

        if not return_stats:
            return f_dom

        # 诊断信息（不参与主训练图）
        with torch.no_grad():
            # filter energy stats: mean/std over batch and spatial
            e32 = e.to(dtype=torch.float32)  # [B,Cf,H,W]
            per_filter_mean = e32.mean(dim=(0, 2, 3))  # [Cf]
            per_filter_std = e32.std(dim=(0, 2, 3), unbiased=False)  # [Cf]

            f32 = f_dom.to(dtype=torch.float32)  # [B,D]
            token_mean = f32.mean(dim=0)  # [D]
            token_std = f32.std(dim=0, unbiased=False)  # [D]
            token_norm = f32.norm(dim=1)  # [B]

            g = self.filter_bank.gain.detach().to(dtype=torch.float32)  # [Cf]
            stats: Dict[str, torch.Tensor] = {
                "energy_per_filter_mean": per_filter_mean,  # [Cf]
                "energy_per_filter_std": per_filter_std,  # [Cf]
                "token_mean": token_mean,  # [D]
                "token_std": token_std,  # [D]
                "token_l2_norm": token_norm,  # [B]
                "gain": g,  # [Cf]
            }

        return f_dom, stats
