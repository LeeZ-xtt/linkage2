# 域编码器改进总结：StyleStatDomainEncoder (方案 A)

## 1. 改进背景与目标
原有的 `DomainEncoder` 是一个基于 BatchNorm 的轻量级 CNN。在少样本域泛化（DG-FSL）任务中，它存在两个主要问题：
1.  **语义泄漏**：全可学习的卷积层容易退化为“域ID分类器”，编码了过多的语义信息（Shortcut），而非纯粹的风格/工况信息。
2.  **统计量不稳定**：BatchNorm 严重依赖 Batch 统计，在 Episodic 训练（N-way K-shot）或多域混合采样时，统计量容易发生偏移，导致训练不稳定。

本次重构的目标是构建一个**显式风格/工况统计编码器**，输出连续的风格坐标 `f_dom`，使其**难以携带语义**，并对 AMP 训练友好。

## 2. 核心架构 Pipeline

新模型 `StyleStatDomainEncoder` 采用固定滤波器组提取特征，并通过显式统计汇聚信息。

> **流程**: 输入 `x` -> **预处理** -> **FixedFilterBank2D** -> **Energy + GN** -> **MultiScaleStatPool** -> **DomainTokenHead** -> 输出 `f_dom`

### 2.1 预处理与去语义
*   **灰度化**：输入图像首先被转换为灰度图 `[B, 1, H, W]`，从源头减少色彩相关的语义干扰。

### 2.2 FixedFilterBank2D (固定滤波器组)
使用一组**固定且不可学习**的经典视觉滤波器，强制模型关注特定的物理/纹理特征：
*   **一阶梯度**：Scharr X/Y (边缘检测)。
*   **二阶导数**：Laplacian (8-neighbor, 细节/噪声检测)。
*   **多尺度 DoG**：Difference of Gaussians (σ=0.6, 1.0, 1.6)，模拟不同频带的带通滤波。
*   **Gabor Bank**：4 个方向 (0°, 45°, 90°, 135°) × 2 个尺度，捕获方向性纹理。

**弱可学习机制**：
*   卷积核权重 (`weight`) 注册为 `buffer`，**完全无梯度**。
*   引入 **Per-Filter Gain** (可学习缩放因子)，允许模型自适应调整不同滤波器的重要性。
*   (默认关闭) 残差微调核 `ΔW`，仅在极特殊情况下开启以适配特殊纹理。

### 2.3 StyleEnergyNonlinearity (能量与归一化)
*   **能量映射**：使用 `sqrt(r^2 + eps)` 计算响应能量。相比 ReLU/Abs，该函数在零点附近更平滑，且针对 AMP (bf16) 进行了数值稳定性优化。
*   **归一化**：全面替换 BatchNorm，使用 **GroupNorm (num_groups=1)**。这等效于对通道维度的 Instance Norm/Layer Norm，不依赖 Batch 统计，保证了单样本推理的一致性。

### 2.4 MultiScaleStatPool (多尺度统计池化)
*   **多尺度视图**：在 Scale 0 (原图), Scale 1 (1/2), Scale 2 (1/4) 三个层级提取统计量。
*   **显式统计量**：
    *   **均值 (Mean)**：反映整体能量强度。
    *   **对数标准差 (Log-Std)**：反映对比度/纹理变化剧烈程度。
*   **低秩 Gram 矩阵** (可选，默认开启)：
    *   在 Scale 0 上，先通过**固定随机投影**将通道压缩至 `Cr=16`。
    *   计算 Gram 矩阵并拉平，捕获全局纹理相关性 (Texture Correlation)。
    *   使用低秩设计严格控制参数量，防止过拟合。

### 2.5 DomainTokenHead (投影头)
*   **结构**：`LayerNorm` -> `Linear` -> `GELU` -> `Dropout` -> `Linear` -> `Tanh` -> `Scale`。
*   **维度压缩**：输出维度从旧版的 128 减少至 **64**，配合 DomainSupCon 进一步投影至 **32**，形成更紧凑的信息瓶颈 (Information Bottleneck)。
*   **数值约束**：输出通过 `Tanh` 和可学习 `Scale` 约束，避免幅值发散。

## 3. 关键配置参数 (`config.py`)

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `domain_dim` | **64** | 域特征输出维度 (原 128) |
| `domain_supcon_proj_dim` | **32** | 对比学习投影维度 (制造瓶颈) |
| `domain_filterbank_kernel` | 7 | 滤波器核大小 |
| `domain_norm` | `"gn"` | 归一化方式 (禁用 BN) |
| `domain_filterbank_trainable_gain` | `True` | 允许学习滤波器增益 |
| `domain_filterbank_trainable_delta_kernel` | `False` | **禁止**学习卷积核微调 (防语义) |
| `domain_use_gram_stats` | `True` | 启用 Gram 纹理统计 |

## 4. 优势与验收标准

1.  **梯度流安全**：固定核无梯度，仅 Gain 和 Head 更新。这从物理上切断了通过卷积核学习高层语义的路径。
2.  **Episodic 友好**：无 BN 设计使得模型在 Meta-Learning 的 Support/Query 划分下表现稳定，不受 Batch Size 影响。
3.  **可解释性 (Diagnosability)**：
    *   支持 `return_stats=True` 接口。
    *   可直接输出各滤波器的能量均值、`gain` 分布。
    *   便于分析模型是依靠“高频噪声”、“垂直纹理”还是“色彩对比度”来区分域。
4.  **AMP 稳定性**：能量计算部分内部使用 FP32 精度，防止 FP16/BF16 下的溢出或精度丢失。

