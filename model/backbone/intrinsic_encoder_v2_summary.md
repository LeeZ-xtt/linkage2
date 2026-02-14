# 本征编码器重构报告：SA-ResNet12 v2

## 1. 概述
本次重构将原有的 ResNet12 本征编码器升级为 **SA-ResNet12 v2 (Shape-Aware + Anti-Alias + Style-Robust)**。

**核心目标**：
1.  **强化结构/语义特征**：通过显式的边缘提取和结构化设计，迫使网络关注物体的形状（Shape）和物理结构（Structure）。
2.  **弱化纹理/风格捷径**：通过抗混叠下采样、混合风格训练和去风格归一化，抑制网络对色彩、光照和局部纹理统计信息的过度依赖。
3.  **提升跨域泛化能力**：在保持 Few-Shot 分类性能的同时，显著提升模型在未见域（Unseen Domains）上的稳定性。

---

## 2. 核心组件详解

### 2.1 FixedSobel Stem (显式边缘先验)
-   **位置**：网络入口处。
-   **机制**：在 RGB 图像输入网络前，并联一个固定的 Sobel 算子分支。
-   **作用**：计算图像的梯度幅值（Edge Magnitude），并将边缘图与原始 RGB 拼接作为初始输入。这为网络提供了一个与色彩无关的“结构强先验”。

### 2.2 Anti-Aliasing Downsampling (BlurPool)
-   **位置**：所有 Stage 的下采样层。
-   **机制**：使用深度可分离的 **Binomial Kernel ([1, 2, 1])** 进行低通滤波，然后再进行 stride=2 的下采样。
-   **作用**：替代传统的 MaxPool 或 Strided Conv。防止高频纹理噪声在下采样过程中产生混叠（Aliasing），从而减少由分辨率变化引起的特征漂移，提升平移不变性。

### 2.3 混合归一化策略 (Hybrid Normalization)
-   **Stage 1 & 2**: 使用 **Instance Normalization (IN)**。
    -   **目的**：在浅层强力去除单张图像的风格统计量（均值/方差），防止低级纹理信息向深层传播。
-   **Stage 3 & 4**: 使用 **GroupNorm (GN)** + **Weight Standardization (WSConv)**。
    -   **目的**：深层特征包含更多语义，过度使用 IN 会破坏语义信息。GN+WS 组合能在不依赖 Batch 统计（适合小 Batch Size 的 Few-Shot 任务）的前提下，提供更稳定的训练梯度和语义一致性。

### 2.4 MixStyle (风格混合)
-   **位置**：Stage 1 与 Stage 2 之间。
-   **机制**：仅在训练期开启。以一定概率（p=0.5）随机混合 Batch 内不同样本的特征统计量（均值和方差）。
-   **作用**：模拟数据增强，生成具有不同风格统计但语义内容不变的新样本，迫使网络学习对风格变化鲁棒的特征。

### 2.5 DropBlock2D (结构化正则)
-   **位置**：Stage 2, 3, 4 的残差块末尾。
-   **机制**：随机丢弃特征图上连续的空间块（Block），而非独立的像素点。
-   **作用**：防止网络过拟合于图像中的某个局部显著特征（如猫的眼睛纹理），强迫网络利用图像的全局上下文信息进行分类。

---

## 3. 架构规格 (Stage-wise Specification)

| 阶段 | 输入通道 | 输出通道 | 关键算子 | 归一化 | 正则化/增强 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stem** | 3 (RGB) | 64 | FixedSobel + Conv3x3 | IN | - |
| **Stage 1** | 64 | 64 | ResBlock + BlurPool | IN | MixStyle (Output) |
| **Stage 2** | 64 | 160 | ResBlock + BlurPool | IN | DropBlock |
| **Stage 3** | 160 | 320 | ResBlock + BlurPool | **GN + WS** | DropBlock |
| **Stage 4** | 320 | **640** | ResBlock (Dilated) | **GN + WS** | DropBlock |

*注：Stage 4 默认开启 `dilated=True`，保持高分辨率特征图，利于原型计算。*

---

## 4. 接口与兼容性

新模块完全兼容旧版 `IntrinsicEncoder` 的接口定义，可无缝替换：

-   **类名**：`IntrinsicEncoder`
-   **输入**：`forward(x)` 接收 `[B, 3, H, W]` 张量。
-   **输出**：返回 `[B, 640, h, w]` 特征图。
-   **属性**：`feature_dim = 640`。
-   **参数**：构造函数支持 `drop_rate` 和 `dilated` 参数，内部自动映射到新的 DropBlock 强度和 Dilation 策略。

---

## 5. 预期效果

1.  **本征特征纯度提升**：`f_phy` 中包含的域信息（Domain Info）将显著减少。
2.  **域泛化能力增强**：在 PACS 等数据集的跨域测试（如 Art -> Sketch）中，准确率预期会有所提升。
3.  **训练稳定性**：虽然引入了 MixStyle 和 DropBlock 等干扰项，但 WSConv 的引入有助于稳定深层梯度，整体收敛曲线可能在初期稍慢，但后期更优。

