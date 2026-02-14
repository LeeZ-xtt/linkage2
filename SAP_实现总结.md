# SAP (Semantic Anchor-driven Purification) 模块实现总结

## 📋 实现概览

已成功实现 SAP 语义锚点驱动净化模块，并完整集成到 ExpB1Model 训练流程中。

## ✅ 完成的工作

### 1. 核心模块实现 (`model/module/semantic_anchor_purification.py`)

**类名**: `SemanticAnchorPurification`

**核心功能**:
- ✅ 步骤 1: 语义锚点（原型）计算
- ✅ 步骤 2: 域特征投影（Query 生成）
- ✅ 步骤 3: 交叉注意力计算
- ✅ 步骤 4: 语义成分重构
- ✅ 步骤 5: 自适应净化强度门控
- ✅ 步骤 6: 残差净化
- ✅ 步骤 7: 正交约束损失计算

**关键参数**:
```python
dom_dim=64          # 域特征维度
phy_dim=640         # 本征语义特征维度
dropout=0.1         # 投影层 dropout
normalize_query=True  # Query L2 归一化
use_layernorm=True   # 输出 LayerNorm
```

**可学习参数**:
- `W_q`: Query 投影 (64→640)
- `W_proj`: 降维投影 (640→64)
- `W_orth`: 正交损失投影 (64→640)
- `alpha`: 净化强度门控（标量）
- `output_ln`: LayerNorm 参数

**初始化策略**:
- Xavier Uniform (gain=0.5) for W_q, W_proj
- Xavier Uniform (gain=1.0) for W_orth
- alpha = 0.0 → sigmoid(0.0) = 0.5

### 2. 模型集成 (`model/exp_b1_model.py`)

**修改点**:
1. ✅ 导入 SAP 模块
2. ✅ 添加 `__init__` 参数: `use_sap`, `sap_dropout`, `sap_orth_weight`, `k_shot`
3. ✅ 实例化 SAP 模块
4. ✅ 在 `forward` 中调用 SAP 净化
5. ✅ 修改 Domain Classifier 和 Domain SupCon 使用净化后的特征
6. ✅ 返回 SAP 损失和统计信息

**集成位置**: 特征提取后、对比学习前

**净化流程**:
```python
# 合并 support + query 特征
all_phy = torch.cat([s_phy, q_phy], dim=0)  # [100, 640]
all_dom = torch.cat([s_dom, q_dom], dim=0)  # [100, 64]
all_labels = torch.cat([support_labels, query_labels], dim=0)

# SAP 净化
all_dom_pure, attn_weights, sap_loss, sap_stats = self.sap_module(
    f_dom=all_dom,
    f_phy=all_phy,
    labels=all_labels,
    n_way=n_way,
    k_shot=self.k_shot
)

# 分离净化后的特征
s_dom = all_dom_pure[:support_size]
q_dom = all_dom_pure[support_size:]
```

### 3. 配置文件 (`config.py`)

**新增配置项**:
```python
# SAP 模块配置
use_sap = True              # 是否使用 SAP
sap_dropout = 0.1           # 投影层 dropout
sap_orth_weight = 0.1       # 正交损失权重
```

### 4. 训练脚本 (`train_b1_improvement.py`)

**修改点**:
1. ✅ 模型创建时传递 SAP 参数
2. ✅ `run_episode` 返回值增加 `sap_loss`, `sap_stats`
3. ✅ 总损失计算添加 SAP 正交损失
4. ✅ NaN 检测添加 `sap_loss` 检查

**总损失公式**:
```python
total_loss = (
    cls_loss +
    λ_dom × domain_loss +
    λ_int_sup × intrinsic_supcon_loss +
    λ_dom_sup × domain_supcon_loss +
    λ_orth × sap_loss  # 新增
)
```

其中 `λ_orth = 0.1`

## 🧪 测试验证

**测试脚本**: `test_sap_module.py`

**测试结果**:
- ✅ 测试 1: 前向传播 - 形状正确，注意力权重和为 1
- ✅ 测试 2: 梯度回传 - 所有参数接收梯度
- ✅ 测试 3: 净化效果 - 余弦相似度降低
- ✅ 测试 4: 模型集成 - 完整前向传播无错误

**关键指标**:
```
净化前余弦相似度: 0.295296
净化后余弦相似度: 0.290205
相似度降低: 0.005091
正交损失: 0.001702
净化强度 (gate): 0.500000
```

## 📊 统计信息输出

SAP 模块返回的 `sap_stats` 字典包含：

| 指标 | 说明 | 期望值 |
|------|------|--------|
| `gate` | 当前净化强度 | 0.0~1.0，训练中自适应 |
| `attn_entropy` | 注意力分布熵 | 越高越均匀 |
| `attn_max` | 最大注意力均值 | 越低越分散 |
| `semantic_norm` | 语义成分范数 | 反映混入程度 |
| `purification_ratio` | 净化比例 | gate × semantic_norm / dom_norm |

## 🎯 设计亮点

1. **不 detach 原型**: 允许 SAP 损失影响本征编码器，增强语义表达
2. **Query 归一化**: 将点积转化为余弦相似度，数值稳定
3. **自适应门控**: 可学习的净化强度，避免过度净化
4. **LayerNorm 输出**: 稳定净化后特征分布
5. **完整梯度流**: SAP → Domain Encoder，实现端到端优化

## 🔧 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `sap_dropout` | 0.1 | 防止过拟合 |
| `sap_orth_weight` | 0.1 | 正交损失权重 |
| `normalize_query` | True | 启用 Query 归一化 |
| `use_layernorm` | True | 启用输出 LayerNorm |
| `temperature` | sqrt(640)≈25.3 | 注意力温度（固定） |

## 📈 预期效果

1. **CDSC-Leak 降低**: 净化后的域特征不再包含类别信息
2. **Domain SupCon 更纯粹**: SDDC 和 CDSC 的对比更加明确
3. **泛化性能提升**: 域特征与语义特征正交，减少域捷径

## 🚀 下一步

1. **训练验证**: 运行完整训练，观察 LeakIndex 曲线
2. **消融实验**: 对比启用/禁用 SAP 的性能差异
3. **可视化分析**: 绘制注意力权重热图，分析净化模式
4. **超参数调优**: 根据训练曲线调整 `sap_orth_weight`

## 📝 使用示例

```python
# 创建模型（启用 SAP）
model = ExpB1Model(
    n_domains=4,
    use_sap=True,
    sap_dropout=0.1,
    sap_orth_weight=0.1,
    k_shot=5
)

# 前向传播
outputs = model(
    support_images, support_labels,
    query_images, n_way,
    query_domain_labels, query_labels, support_domain_labels
)

# 解包输出（包含 SAP 损失）
(logits, prototypes, domain_logits,
 intrinsic_supcon_loss, intrinsic_supcon_stats,
 domain_supcon_loss, domain_supcon_stats,
 sap_loss, sap_stats) = outputs

# 计算总损失
total_loss = (
    cls_loss +
    0.2 * domain_loss +
    0.3 * intrinsic_supcon_loss +
    0.15 * domain_supcon_loss +
    0.1 * sap_loss  # SAP 正交损失
)
```

---

**实现日期**: 2026-02-11  
**实现者**: Kiro AI Assistant  
**状态**: ✅ 完成并通过测试
