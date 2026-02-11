## 🤖 Assistant

# SAP (Semantic Anchor-driven Purification) 模块完整详细方案

---

## 一、模块概述

### 1.1 模块命名与文件位置
- **模块名称**：`SemanticAnchorPurification` (简称 SAP)
- **文件路径**：`model/module/semantic_anchor_purification.py`
- **类名**：`SemanticAnchorPurification`

### 1.2 设计目标
SAP 模块旨在通过交叉注意力机制，利用本征支（Intrinsic Branch）计算的类别原型作为"语义锚点"，显式识别并去除域特征 `f_dom` 中混入的类别语义成分，输出净化后的 `f_dom_pure`。净化后的域特征应仅编码风格/纹理/域统计信息，与语义特征正交，从而解决训练曲线中观察到的 CDSC-Leak 反弹问题。

### 1.3 理论基础
- **交叉注意力机制**：借鉴 Transformer 中 Query-Key-Value 的注意力计算范式 [^1][^4]，将域特征作为 Query，语义原型作为 Key/Value，计算域特征对各类别语义的"关注程度"
- **子空间投影**：通过注意力加权重构的语义成分代表了域特征中"不应存在"的类别信息，将其减除实现净化
- **残差净化**：采用可学习的净化强度门控，实现软净化而非硬截断，保留对域判别有益的信息

---

## 二、输入输出规格（严格对齐现有代码）

### 2.1 输入张量

| 参数名 | 形状 | 数据类型 | 来源 | 说明 |
|--------|------|----------|------|------|
| `f_dom` | `(B, 64)` | `torch.float32` | `StyleStatDomainEncoder` 输出 | 原始域特征，B = N×(K+Q) |
| `f_phy` | `(B, 640)` | `torch.float32` | `IntrinsicEncoder` 输出 | 本征语义特征 |
| `labels` | `(B,)` | `torch.long` | Episode 数据 | 类别标签，值域 [0, n_way-1] |
| `n_way` | `int` | - | Config | 类别数（通常为 5） |
| `k_shot` | `int` | - | Config | 每类支持样本数（通常为 5） |

**维度计算示例**（以 5-way 5-shot 15-query 为例）：
- Support Set: 5 × 5 = 25 样本
- Query Set: 5 × 15 = 75 样本
- 总计 B = 100 样本

### 2.2 输出张量

| 输出名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `f_dom_pure` | `(B, 64)` | `torch.float32` | 净化后的域特征 |
| `attn_weights` | `(B, n_way)` | `torch.float32` | 交叉注意力权重（用于可视化） |
| `loss_orth` | `scalar` | `torch.float32` | 正交约束损失 |
| `purification_stats` | `dict` | - | 诊断统计信息 |

---

## 三、详细算法流程

### 3.1 步骤 1：语义锚点（原型）计算

**目的**：基于当前 Episode 的 Support Set 计算 N 个类别原型，作为语义锚点。

**算法**：
```
输入: f_phy (B, 640), labels (B,), n_way, k_shot

1. 确定 Support Set 的索引范围:
   support_size = n_way × k_shot
   f_phy_support = f_phy[:support_size]  # (N×K, 640)
   labels_support = labels[:support_size]  # (N×K,)

2. 对每个类别 c ∈ [0, n_way-1]:
   mask_c = (labels_support == c)  # 布尔掩码
   f_phy_c = f_phy_support[mask_c]  # (K, 640)
   prototype_c = f_phy_c.mean(dim=0)  # (640,)

3. 堆叠所有原型:
   prototypes = stack([prototype_0, ..., prototype_{n_way-1}])  # (n_way, 640)

4. L2 归一化原型（提升注意力计算稳定性）:
   prototypes_normalized = F.normalize(prototypes, p=2, dim=1)  # (n_way, 640)

输出: prototypes_normalized (n_way, 640)
```

**实现注意事项**：
- 使用 `torch.zeros` 预分配原型张量，避免动态 list 拼接
- 对 `prototypes` 执行 `detach()` 阻断到本征编码器的梯度（可选，取决于是否希望原型计算影响本征支）

### 3.2 步骤 2：域特征投影（Query 生成）

**目的**：将 64 维域特征投影到 640 维语义空间，作为交叉注意力的 Query。

**算法**：
```
输入: f_dom (B, 64)

1. 线性投影:
   Q = W_q(f_dom)  # W_q: Linear(64, 640, bias=True)
   # Q 形状: (B, 640)

2. 可选：对 Q 进行 L2 归一化（提升点积注意力稳定性）:
   Q_normalized = F.normalize(Q, p=2, dim=1)  # (B, 640)

输出: Q_normalized (B, 640)
```

**设计选择**：
- **使用 bias**：允许投影学习偏移，增加表达能力
- **归一化**：推荐启用，将点积转化为余弦相似度，数值范围可控 [-1, 1]

### 3.3 步骤 3：交叉注意力计算

**目的**：计算每个样本的域特征与各类别语义原型的相似度，得到注意力权重。

**算法**：
```
输入: Q (B, 640), prototypes (n_way, 640)

1. 计算注意力 logits（缩放点积）:
   # Q: (B, 640), prototypes.T: (640, n_way)
   attn_logits = Q @ prototypes.T  # (B, n_way)
   
2. 温度缩放:
   temperature = sqrt(640) = 25.3  # 标准 Transformer 缩放因子
   attn_logits_scaled = attn_logits / temperature  # (B, n_way)
   
3. Softmax 归一化:
   attn_weights = F.softmax(attn_logits_scaled, dim=-1)  # (B, n_way)
   # 每行和为 1，表示该样本对各类别原型的注意力分布

输出: attn_weights (B, n_way)
```

**注意力权重解释**：
- `attn_weights[i, c]` 表示第 i 个样本的域特征与第 c 类语义原型的相关程度
- 高相关性意味着该样本的域特征中混入了第 c 类的语义信息，需要被净化去除

### 3.4 步骤 4：语义成分重构

**目的**：基于注意力权重，从语义原型中聚合出"应被去除的语义成分"。

**算法**：
```
输入: attn_weights (B, n_way), prototypes (n_way, 640)

1. 加权聚合语义成分（在 640 维空间）:
   # attn_weights: (B, n_way), prototypes: (n_way, 640)
   semantic_component_640 = attn_weights @ prototypes  # (B, 640)
   # 每个样本得到一个 640 维向量，代表其域特征中混入的语义信息

2. 投影回域特征空间（降维）:
   semantic_component_64 = W_proj(semantic_component_640)  # W_proj: Linear(640, 64, bias=True)
   # semantic_component_64 形状: (B, 64)

输出: semantic_component_64 (B, 64)
```

**设计理由**：
- 在 640 维空间进行语义聚合，保留完整的语义表达
- 降维回 64 维，实现残差减除

### 3.5 步骤 5：自适应净化强度门控

**目的**：通过可学习参数控制净化强度，避免过度净化导致域信息丢失。

**算法**：
```
输入: 可学习参数 alpha (标量, 初始化为 0.5)

1. Sigmoid 约束到 [0, 1] 范围:
   gate = torch.sigmoid(alpha)  # 标量
   
2. 可选：温度控制 Sigmoid 的陡峭程度:
   gate = torch.sigmoid(alpha * temperature_gate)  # temperature_gate 默认为 1.0

输出: gate (标量, 范围 [0, 1])
```

**门控语义**：
- `gate ≈ 0`：几乎不净化，保留原始域特征
- `gate ≈ 1`：完全净化，最大程度去除语义成分
- 训练过程中，模型会自动学习最优的净化强度

### 3.6 步骤 6：残差净化

**目的**：从原始域特征中减去经门控调制的语义成分，得到净化后的域特征。

**算法**：
```
输入: f_dom (B, 64), semantic_component_64 (B, 64), gate (标量)

1. 残差减除:
   f_dom_pure = f_dom - gate * semantic_component_64  # (B, 64)
   
2. 可选：LayerNorm 稳定输出分布（推荐启用）:
   f_dom_pure = LayerNorm(f_dom_pure)  # LayerNorm(64)

输出: f_dom_pure (B, 64)
```

**LayerNorm 的作用**：
- 稳定净化后特征的数值范围，避免因净化强度变化导致的分布漂移
- 与后续 Domain SupCon 的温度参数协同工作

### 3.7 步骤 7：正交约束损失计算

**目的**：强制净化后的域特征与本征语义特征正交，量化解耦程度。

**算法**：
```
输入: f_dom_pure (B, 64), f_phy (B, 640)

1. 将 f_dom_pure 投影到语义空间（用于计算正交性）:
   f_dom_pure_projected = W_orth(f_dom_pure)  # W_orth: Linear(64, 640, bias=False)
   # 形状: (B, 640)
   
2. 计算余弦相似度:
   cos_sim = F.cosine_similarity(f_dom_pure_projected, f_phy.detach(), dim=1)  # (B,)
   # detach f_phy 防止损失影响本征支的主要优化目标
   
3. 计算正交损失（余弦相似度的平方均值）:
   loss_orth = (cos_sim ** 2).mean()  # 标量
   # 目标：cos_sim → 0，即 f_dom_pure 与 f_phy 正交

输出: loss_orth (标量)
```

**损失设计说明**：
- 使用平方而非绝对值，对大偏差施加更强惩罚
- 对 `f_phy` 执行 `detach()`，确保正交损失不干扰本征支的分类学习

---

## 四、可学习参数与初始化策略

### 4.1 参数列表

| 参数名 | 形状 | 初始化方法 | 说明 |
|--------|------|------------|------|
| `W_q.weight` | `(640, 64)` | Xavier Uniform, gain=0.5 | Query 投影矩阵 |
| `W_q.bias` | `(640,)` | Zeros | Query 投影偏置 |
| `W_proj.weight` | `(64, 640)` | Xavier Uniform, gain=0.5 | 降维投影矩阵 |
| `W_proj.bias` | `(64,)` | Zeros | 降维投影偏置 |
| `W_orth.weight` | `(640, 64)` | Xavier Uniform, gain=1.0 | 正交损失投影（无 bias） |
| `alpha` | `(1,)` | Constant(0.0)→sigmoid=0.5 | 净化强度门控 |
| `output_ln.weight` | `(64,)` | Ones | LayerNorm 缩放 |
| `output_ln.bias` | `(64,)` | Zeros | LayerNorm 偏移 |

### 4.2 初始化代码逻辑（文字描述）

```
对于 W_q 和 W_proj:
  使用 nn.init.xavier_uniform_(weight, gain=0.5)
  # gain=0.5 使初始注意力分布较为平滑，避免训练初期过于尖锐的注意力
  
对于 W_orth:
  使用 nn.init.xavier_uniform_(weight, gain=1.0)
  # 标准初始化，因为仅用于损失计算
  
对于 alpha:
  初始化为 0.0，经 sigmoid 后为 0.5
  # 表示初始净化强度为 50%
  
对于 LayerNorm:
  使用 PyTorch 默认初始化（weight=1, bias=0）
```

---

## 五、损失函数设计

### 5.1 SAP 模块内部损失

**正交约束损失 `L_orth`**：
$$L_{orth} = \frac{1}{B} \sum_{i=1}^{B} \left( \cos(f_{dom}^{pure(i)}, f_{phy}^{(i)}) \right)^2$$

### 5.2 与总损失的集成

在 `train_b1_improvement.py` 中，SAP 的损失应加入总损失：

```
L_total = L_cls (分类损失)
        + λ_dom × L_domain_ce (域分类损失)
        + λ_int × L_intrinsic_supcon (本征支对比损失)
        + λ_dom_sup × L_domain_supcon (域支对比损失，使用 f_dom_pure)
        + λ_orth × L_orth (SAP 正交损失)
```

**推荐权重**：
- `λ_orth = 0.1`
---

## 六、与现有代码的集成

### 6.1 集成位置

**文件**：`model/exp_b1_model.py`

**插入点**：在 `forward` 函数中，特征提取之后、对比学习之前

### 6.2 集成逻辑（文字描述）

```
在 ExpB1Model.__init__ 中:
  1. 导入 SAP 模块
  2. 实例化 self.sap_module = SemanticAnchorPurification(
       dom_dim=64,
       phy_dim=640,
       n_way=config.n_way,  # 可选，运行时传入也可
       dropout=0.1
     )

在 ExpB1Model.forward 中:
  1. 提取特征（现有逻辑不变）:
     s_phy, s_dom = self.backbone(support_images)
     q_phy, q_dom = self.backbone(query_images)
     
  2. 合并 support 和 query:
     all_phy = torch.cat([s_phy, q_phy], dim=0)
     all_dom = torch.cat([s_dom, q_dom], dim=0)
     all_labels = torch.cat([support_labels, query_labels], dim=0)
     
  3. 调用 SAP 模块（新增）:
     all_dom_pure, attn_weights, loss_orth, sap_stats = self.sap_module(
         f_dom=all_dom,
         f_phy=all_phy,
         labels=all_labels,
         n_way=n_way,
         k_shot=self.k_shot  # 需要在 __init__ 中保存
     )
     
  4. 分离 support 和 query 的净化特征:
     support_size = n_way * k_shot
     s_dom_pure = all_dom_pure[:support_size]
     q_dom_pure = all_dom_pure[support_size:]
     
  5. 后续使用 s_dom_pure 和 q_dom_pure 替代 s_dom 和 q_dom:
     - Domain Classifier: domain_logits = self.domain_classifier(all_dom_pure)
     - Domain SupCon: domain_supcon_loss = self.domain_supcon_module(all_dom_pure, ...)
     
  6. 返回值中添加 loss_orth 和 sap_stats
```

### 6.3 对 Domain SupCon 的修改

**修改点**：将输入从 `all_dom` 改为 `all_dom_pure`

**理由**：净化后的域特征不再包含语义信息，Domain SupCon 的正负样本定义更加纯粹：
- SDDC（同域异类）：现在真正衡量的是"风格相似性"
- CDSC（跨域同类）：应该被推远，因为唯一的共同点（类别）已被去除

---

## 七、超参数设置（基于深度学习最佳实践）

### 7.1 固定超参数（无需调整）

| 超参数 | 值 | 依据 |
|--------|-----|------|
| `temperature` (注意力缩放) | `sqrt(640) ≈ 25.3` | Transformer 标准做法 [^1][^4] |
| `alpha` 初始值 | `0.0` (sigmoid=0.5) | 中等净化强度起步 |
| `LayerNorm eps` | `1e-5` | PyTorch 默认值 [^1] |
| `W_q`, `W_proj` 初始化 gain | `0.5` | 避免初始注意力过于尖锐 |

### 7.2 可调超参数（推荐值）

| 超参数 | 推荐值 | 说明 |
|--------|--------|----------|------|
| `λ_orth` (正交损失权重) | `0.1` ||
| `dropout` (投影层) | `0.1` |防止过拟合 |
| `normalize_query` | `True` |是否归一化 Query |
| `use_layernorm` | `True`  | 是否对输出做 LN |


