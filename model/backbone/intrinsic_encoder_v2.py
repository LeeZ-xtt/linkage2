"""
本征编码器 v2 (Intrinsic Encoder v2) - SA-ResNet12 v2
Shape-Aware + Anti-Alias + Style-Robust

设计目标:
1. 强化结构/语义/物理模式 (Shape/Edge/Ridge)
2. 弱化纹理/色彩/风格统计 (Style/Texture Shortcut)
3. 保持与旧接口兼容: forward(x) -> [B, 640, h, w]

核心组件:
- FixedSobel Stem: 引入显式边缘先验
- BlurPool: 抗混叠下采样
- WSConv2d + GN: 替代深层 BN/IN，提升语义一致性
- MixStyle: 训练期风格扰动
- DropBlock: 结构化正则
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type


# ==============================================================================
# 1. 基础算子与辅助模块 (Fixed Operators & Helpers)
# ==============================================================================

class FixedSobel(nn.Module):
    """
    固定 Sobel 算子，用于提取边缘梯度幅值。
    输入: RGB [B, 3, H, W]
    输出: Edge Magnitude [B, 1, H, W]
    """
    def __init__(self):
        super().__init__()
        # 灰度化系数
        self.register_buffer('rgb2gray', torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))
        
        # Sobel 核 (X 和 Y 方向)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 灰度化
        x_gray = (x * self.rgb2gray).sum(dim=1, keepdim=True) # [B, 1, H, W]
        
        # 2. 卷积计算梯度 (Padding=1 保持尺寸)
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        # 3. 梯度幅值
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return magnitude


class BlurPool2d(nn.Module):
    """
    Anti-Aliasing 下采样 (Low-pass Filter + Downsampling)
    使用 Binomial Kernel [1, 2, 1]
    """
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.stride = stride
        
        # Binomial Kernel [1, 2, 1] -> 2D: [[1,2,1], [2,4,2], [1,2,1]] / 16
        k = torch.tensor([1., 2., 1.])
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        k = k.view(1, 1, 3, 3)
        # 扩展到多通道 (Depthwise)
        self.register_buffer('kernel', k.repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise Conv
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.channels)


class WSConv2d(nn.Conv2d):
    """
    Weight Standardization Conv2d
    在卷积前对权重进行标准化: (w - mean) / (std + eps)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                         dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight_std = weight.std(dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (weight - weight_mean) / (weight_std + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)


class MixStyle(nn.Module):
    """
    MixStyle: 在特征统计层面混合风格 (仅训练期)
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        # 生成混合系数 lambda
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        
        # 随机 shuffle batch
        perm = torch.randperm(B).to(x.device)
        
        # 计算统计量
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        
        mu_perm = mu[perm]
        sig_perm = sig[perm]
        
        # 混合统计量
        mu_mix = mu * lmda + mu_perm * (1 - lmda)
        sig_mix = sig * lmda + sig_perm * (1 - lmda)
        
        # 应用新风格: Norm -> Scale -> Shift
        return ((x - mu) / sig) * sig_mix + mu_mix


class DropBlock2D(nn.Module):
    """
    DropBlock: 空间连续遮挡正则化
    """
    def __init__(self, drop_prob: float = 0.1, block_size: int = 5):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
            
        # Gamma: 根据 block_size 调整丢弃概率，使其符合期望的 drop_prob
        # gamma = (drop_prob * H * W) / (block_size^2 * (H - block_size + 1) * (W - block_size + 1))
        # 简化计算，假设 H, W 足够大
        N, C, H, W = x.shape
        gamma = self.drop_prob * (H * W) / (self.block_size ** 2) / ((H - self.block_size + 1) * (W - self.block_size + 1))
        
        # 生成 mask center
        mask = (torch.rand(N, 1, H - self.block_size + 1, W - self.block_size + 1, device=x.device) < gamma).float()
        
        # 扩展 mask block (Max Pool 近似膨胀)
        # Padding 保证尺寸一致
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        
        # 裁剪回原始尺寸 (padding 可能导致微小尺寸变化)
        mask = mask[:, :, :H, :W]
        
        # 反转 mask (1 为保留，0 为丢弃)
        mask = 1 - mask
        
        # 归一化以保持激活期望值
        return x * mask * (mask.numel() / (mask.sum() + 1e-6))


# ==============================================================================
# 2. ResNet12 核心组件 (Blocks)
# ==============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, use_ws: bool = False) -> nn.Module:
    """3x3 卷积，可选 WSConv2d"""
    conv_cls = WSConv2d if use_ws else nn.Conv2d
    return conv_cls(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, use_ws: bool = False) -> nn.Module:
    """1x1 卷积"""
    conv_cls = WSConv2d if use_ws else nn.Conv2d
    return conv_cls(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet12Block(nn.Module):
    """
    ResNet12 基础块 (3个卷积层)
    结构:
      Conv3x3 -> Norm -> LeakyReLU
      Conv3x3 -> Norm -> LeakyReLU
      Conv3x3 -> Norm
      Add Residual
      LeakyReLU
      (Optional) Downsample via BlurPool
      (Optional) DropBlock
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Type[nn.Module] = nn.InstanceNorm2d,
        use_ws: bool = False,
        drop_block: Optional[nn.Module] = None,
        dilation: int = 1,
        keep_resolution: bool = False # 如果为 True，即使 stride=2 也不下采样 (用于 dilated stage)
    ) -> None:
        super().__init__()
        
        # 3个卷积层，stride 始终为 1 (ResNet12 特性：下采样在块末尾)
        self.conv1 = conv3x3(inplanes, planes, dilation=dilation, use_ws=use_ws)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = conv3x3(planes, planes, dilation=dilation, use_ws=use_ws)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = conv3x3(planes, planes, dilation=dilation, use_ws=use_ws)
        self.bn3 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.drop_block = drop_block
        
        # 下采样策略：如果 stride > 1，使用 BlurPool
        # 注意：如果 keep_resolution=True (dilation 模式)，则不使用 BlurPool
        self.blur_pool = None
        if stride > 1 and not keep_resolution:
            self.blur_pool = BlurPool2d(planes, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 注意：当 stride>1 且启用 BlurPool 时，shortcut 分支会在 downsample 内下采样。
        # 为保证残差相加形状一致，主分支也必须在相加前完成同样的下采样。
        if self.blur_pool is not None:
            out = self.blur_pool(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
            
        # 应用 DropBlock
        if self.drop_block is not None:
            out = self.drop_block(out)

        return out


# ==============================================================================
# 3. 主本征编码器 (Main Intrinsic Encoder)
# ==============================================================================

class IntrinsicEncoder(nn.Module):
    """
    本征编码器 v2 (SA-ResNet12 v2)
    
    参数:
        drop_rate: DropBlock/Dropout 强度 (默认 0.2)
        dilated: 是否在 Stage 4 使用空洞卷积保持分辨率 (默认 True)
        use_sobel: 是否启用 Sobel Stem (默认 True)
        use_blurpool: 是否启用 Anti-aliasing (默认 True)
        use_mixstyle: 是否启用 MixStyle (默认 True)
        use_wsconv: 是否启用 Weight Standardization (默认 True, 仅 Stage 3/4)
    """
    def __init__(
        self,
        drop_rate: float = 0.2,
        dilated: bool = True,
        use_sobel: bool = True,
        use_blurpool: bool = True,
        use_mixstyle: bool = True,
        use_wsconv: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.feature_dim = 640
        self.dilated = dilated
        self.drop_rate = drop_rate
        
        # 开关配置
        self.config = {
            'use_sobel': use_sobel,
            'use_blurpool': use_blurpool,
            'use_mixstyle': use_mixstyle,
            'use_wsconv': use_wsconv,
            'norm_stage12': 'IN',
            'norm_stage34': 'GN'
        }

        # 1. Stem (Input -> Stage 1)
        # ---------------------------------------------------------
        self.sobel = FixedSobel() if use_sobel else None
        # 输入通道: RGB(3) + Sobel(1) = 4 (如果启用 Sobel)
        in_channels = 4 if use_sobel else 3
        
        # 初始映射 (为了匹配 standard ResNet12 的 64 通道起步)
        # 使用 3x3 小卷积 + IN + LeakyReLU
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.inplanes = 64
        
        # 2. Stages
        # ---------------------------------------------------------
        
        # Stage 1: 64 ch, IN, No WS
        self.layer1 = self._make_layer(
            planes=64, stride=2, 
            norm_type='IN', use_ws=False, 
            drop_prob=0.0 # Stage 1 通常不加 DropBlock
        )
        
        # MixStyle 插入点 (Stage 1 后 / Stage 2 前)
        self.mixstyle = MixStyle(p=0.5, alpha=0.1) if use_mixstyle else nn.Identity()
        
        # Stage 2: 160 ch, IN, No WS, DropBlock
        self.layer2 = self._make_layer(
            planes=160, stride=2, 
            norm_type='IN', use_ws=False, 
            drop_prob=min(0.1, drop_rate) # 轻微 DropBlock
        )
        
        # Stage 3: 320 ch, GN, WS, DropBlock
        self.layer3 = self._make_layer(
            planes=320, stride=2, 
            norm_type='GN', use_ws=use_wsconv, 
            drop_prob=min(0.15, drop_rate) # 中等 DropBlock
        )
        
        # Stage 4: 640 ch, GN, WS, DropBlock, Optional Dilation
        # 如果 dilated=True, stride=1, dilation=2
        last_stride = 1 if dilated else 2
        last_dilation = 2 if dilated else 1
        
        self.layer4 = self._make_layer(
            planes=640, stride=last_stride, 
            norm_type='GN', use_ws=use_wsconv, 
            drop_prob=min(0.15, drop_rate),
            dilation=last_dilation,
            keep_resolution=dilated
        )
        
        # 初始化权重
        self._init_weights()

    def _make_layer(
        self, 
        planes: int, 
        stride: int, 
        norm_type: str, 
        use_ws: bool, 
        drop_prob: float,
        dilation: int = 1,
        keep_resolution: bool = False
    ) -> nn.Sequential:
        """构建 ResNet Stage"""
        
        # 归一化层选择
        if norm_type == 'IN':
            def norm_layer(c): return nn.InstanceNorm2d(c, affine=True)
        elif norm_type == 'GN':
            # GroupNorm: Group 数设为 32，如果通道少于 32 则设为通道数的一半或更小
            def norm_layer(c): 
                groups = 32
                if c < 32: groups = 4
                return nn.GroupNorm(groups, c, affine=True)
        else:
            def norm_layer(c): return nn.BatchNorm2d(c)

        downsample = None
        # 下采样路径逻辑 (Shortcut)
        # 如果通道数变化 或 stride != 1，需要 downsample
        if stride != 1 or self.inplanes != planes:
            # 路径: Conv1x1 -> Norm -> (Optional BlurPool)
            layers = [
                conv1x1(self.inplanes, planes, stride=1, use_ws=use_ws), # 先升维/变维
                norm_layer(planes)
            ]
            if stride > 1 and not keep_resolution:
                # 使用 BlurPool 下采样
                if self.config['use_blurpool']:
                    layers.append(BlurPool2d(planes, stride=stride))
                else:
                    layers.append(nn.AvgPool2d(stride)) # 回退到 AvgPool
            
            downsample = nn.Sequential(*layers)

        # DropBlock
        drop_block = None
        if drop_prob > 0 and self.config.get('use_dropblock', True): # 默认开启 DropBlock
             drop_block = DropBlock2D(drop_prob=drop_prob, block_size=5)

        block = ResNet12Block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            norm_layer=norm_layer,
            use_ws=use_ws,
            drop_block=drop_block,
            dilation=dilation,
            keep_resolution=keep_resolution
        )

        self.inplanes = planes
        return block # ResNet12 每个 stage 只有一个 Block (含 3 个 conv)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, WSConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            feature_map: [B, 640, h, w]
        """
        # 1. Stem
        if self.sobel is not None:
            edge = self.sobel(x)
            x = torch.cat([x, edge], dim=1) # [B, 4, H, W]
        
        x = self.stem_conv(x)
        
        # 2. Stages
        x = self.layer1(x)
        
        # MixStyle (仅训练期)
        x = self.mixstyle(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def get_diagnostics(self) -> dict:
        """返回当前配置诊断信息"""
        return {
            'drop_rate': self.drop_rate,
            'dilated': self.dilated,
            **self.config
        }

