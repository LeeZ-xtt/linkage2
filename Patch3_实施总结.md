# Patch 3 实施总结

## 目标

让 SAP 的 cross-attention **从"均匀撒糖"变成"会偏向某些原型"**，同时避免 scale 越学越大导致数值不稳、梯度暴躁。

## 核心改动

### 1. SAP 模块 (`model/module/semantic_anchor_purification.py`)

#### 1.1 `__init__` 方法
- ❌ 移除：`self.temperature = phy_dim ** 0.5`（固定温度）
- ✅ 新增：
  ```python
  logit_scale_init = 25.0
  self.logit_scale_log = nn.Parameter(torch.log(torch.tensor(logit_scale_init)))
  self.logit_scale_max = 100.0
  ```
  - 用 log 参数化保证 scale > 0
  - 初始值 25（与原 temperature ≈25.3 接近）
  - 上限 100（参考 CLIP，防止 attention 过尖）

#### 1.2 `compute_cross_attention` 方法
- ❌ 旧逻辑：`attn_logits / self.temperature`（压扁 logits）
- ✅ 新逻辑：
  ```python
  scale = torch.exp(self.logit_scale_log).clamp(max=self.logit_scale_max)
  attn_logits_scaled = attn_logits * scale
  # 数值稳定性：row-wise 减最大值
  attn_logits_scaled = attn_logits_scaled - attn_logits_scaled.max(dim=-1, keepdim=True)[0]
  ```
  - 可学习的"音量旋钮"
  - 封顶防止发散
  - 减最大值防止数值溢出

#### 1.3 `forward` 方法（stats 监控）
- ✅ 新增监控指标：
  ```python
  'logit_scale': current_scale.item(),  # 当前 scale 值
  'logit_scale_clipped': scale_clipped  # 触顶比例
  ```

### 2. ExpB1Model (`model/exp_b1_model.py`)

#### 2.1 `get_parameters` 方法
- ❌ 旧逻辑：`return list(self.parameters())`（统一 weight_decay）
- ✅ 新逻辑：
  ```python
  # 对 logit_scale_log 禁用 weight_decay
  no_decay_params = [self.sap_module.logit_scale_log]
  decay_params = [其他所有参数]
  
  return [
      {'params': decay_params},  # 使用默认 weight_decay=5e-4
      {'params': no_decay_params, 'weight_decay': 0.0}  # 禁用 weight_decay
  ]
  ```
  - 防止 weight_decay 把 log_scale 往 0 拉（让 scale 往 1 靠）

### 3. 训练代码 (`train_b1_improvement.py`)

#### 3.1 参数组处理
- ✅ 修改：
  ```python
  # 获取参数组（支持分组）
  param_groups = model.get_parameters()
  optimizer = torch.optim.SGD(param_groups, ...)
  
  # 提取所有参数用于梯度裁剪
  all_params = []
  for group in param_groups:
      all_params.extend(group['params'])
  
  # 梯度裁剪时使用 all_params
  torch.nn.utils.clip_grad_norm_(all_params, max_norm=Config.grad_clip_norm)
  ```
  - 兼容参数组格式
  - 正确处理梯度裁剪

## 验证结果

### ✅ 功能验证 (`verify_patch3.py`)
- logit_scale_log 参数存在且为 Parameter
- 初始 scale = 25.00（符合预期）
- scale 上限 = 100.0
- temperature 已成功移除
- 前向传播正常：
  - 注意力熵 = 1.38 < log(5) = 1.61（不再均匀）
  - 注意力最大值 = 0.42 > 1/5 = 0.20（会偏向某些原型）
- clamp 功能正常（设置 200 后被限制到 100）

### ✅ 参数分组验证 (`verify_optimizer.py`)
- 参数组 0：82 个参数，使用默认 weight_decay
- 参数组 1：1 个参数（logit_scale_log），weight_decay=0.0

## 预期训练效果

### 监控指标（从 stats 获取）
- `attn_entropy`：应明显低于 `log(n_way)`（不再均匀）
- `attn_max`：应明显高于 `1/n_way`（会偏向某些原型）
- `logit_scale`：
  - 初期：≈25
  - 训练中：可能上下调整（15~30 范围内）
  - 不应长期贴着 100（贴着说明原型区分度不足）
- `logit_scale_clipped`：应接近 0（很少触顶）

### 与 Patch 2 的对比
- Patch 2（固定 temperature）：attention 过于均匀，SAP 净化效果弱
- Patch 3（可学习 scale）：attention 会自适应调整，偏向相关原型

## 文件清单

### 修改的文件
1. `model/module/semantic_anchor_purification.py`
   - `__init__`：新增 logit_scale_log 参数
   - `compute_cross_attention`：改用可学习 scale
   - `forward`：新增 stats 监控

2. `model/exp_b1_model.py`
   - `get_parameters`：对 logit_scale_log 禁用 weight_decay

3. `train_b1_improvement.py`
   - 修改参数组处理逻辑
   - 修复梯度裁剪兼容性

### 新增的文件
1. `verify_patch3.py`：功能验证脚本
2. `verify_optimizer.py`：参数分组验证脚本
3. `Patch3_实施总结.md`：本文档

## 下一步

1. 运行完整训练，观察 `logit_scale` 的演化曲线
2. 对比 Patch 2 vs Patch 3 的验证精度
3. 如果 `logit_scale` 长期贴着 100，考虑：
   - 增大 `logit_scale_max`（如 200）
   - 或检查原型质量（可能需要调整 SAP 其他超参数）

## 技术细节

### 为什么用 log 参数化？
- 保证 scale > 0（exp 永远为正）
- 优化器更新时更稳定（log 空间的梯度更平滑）

### 为什么要 clamp？
- 防止 scale 过大导致 softmax 过尖（梯度消失）
- 防止数值溢出（exp(大数) 会爆炸）

### 为什么要 row-wise 减最大值？
- softmax 的数值稳定性技巧（标准做法）
- 不改变 softmax 结果，但避免 exp(大数) 溢出

### 为什么 logit_scale_log 不用 weight_decay？
- weight_decay 会把参数往 0 拉
- log_scale → 0 意味着 scale → 1
- scale=1 时 attention 又会变软（回到 Patch 2 的问题）
