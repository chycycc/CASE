# CASE-EPCL V9 Trial 9 架构设计与执行计划：原型自适应软记忆反哺（升级版 PCAM）

> **文档性质**：独立实施技术方案与执行计划（供后续会话与其他 AI 严格逐步执行）。  
> **前置依赖**：Trial 8 训练运行完毕并完成指标基准测定。  
> **核心任务**：彻底剥离解码端加性词表偏置代码，激活并升级原型交叉记忆注意力模块（PCAM），建立连续隐层情感先验反哺机制。  
> **代码分支**：`v9-clean`

---

## 一、背景与前置状态

1. **当前阶段（Trial 8）**：
   - 正在测定 `ACCUM_STEPS=1`（单步物理 32 零累加，更新次数 10,000 次）在基准架构下的性能上限。
   - 基准参照为 Trial 7（Test PPL 33.58，Test EMO_acc 42.25%，Alignment 0.9353）。
2. **Trial 9 定位**：
   - 在 Trial 8 确立的步频基线上，实施单一维度的模型结构重构。
   - 解决的核心矛盾：**收复自回归语言建模低困惑度（PPL 32.7x），同时避免概率分布过度锐化引发的束搜索（Beam Search）多样性坍塌**。

---

## 二、候选方案科学论证与排除记录

在方案设计阶段，对另外两个潜在方向进行了理论推演与工程核验，均予以否决：

### 1. 否决方向 1：双曲几何流形原型学习（Poincaré Ball）
- **标签拓扑不匹配**：双曲空间对层次树结构（DAG）具有低失真嵌入特性，但实验数据（EmpatheticDialogues）为扁平的 32 类单标签分布（1-of-32），无先验层级监督。在无显式层级约束下，特征向庞加莱球边界（$r \to 1 - \epsilon$）扩散，存在表征塌陷与浮点溢出风险。
- **算子与计算图冲突**：当前架构依赖动量样本中心原型（MCP）。双曲空间质心为弗雷歇均值（Fréchet Mean），无闭式解析解，必须通过黎曼梯度多步迭代求解，计算吞吐低且反向传播数值不稳定。
- **解码器跨流形失真**：自回归解码器自注意力工作于欧几里得切空间，双曲表征投回切空间会引入流形失真，对 Test PPL 带来负面风险。

### 2. 否决方向 2：语义-情感互信息最小化解耦（$I(z_{sem}; z_{emo}) \to 0$）
- **与基础机制冲突**：CASE 采用认知图与概念图双流结构，并使用互信息最大化（MIM）对齐事实逻辑与情感倾向。自然语言中事实逻辑与主观情绪存在因果伴生关系，强行约束独立会削弱解码器生成线索，直接损害语言连贯性。
- **多任务优化失衡**：互信息估计（对抗判别器或 CLUB 上界）方差大，与 NLL、BOW、MIM、EPCL 等损失叠加后，梯度冲突超出 PCGrad 的调节范围。

---

## 三、Trial 9 核心机制：升级版 PCAM 原型软记忆反哺

### 1. 机制重构对比

```
[历史设计：词表层加性偏置 (已暴露缺陷)]
dec_hidden ──> Generator ──> Logits ──+──> (emo_to_vocab * gate * mask) ──> Final Logits
                                      ↑
                              硬性词表偏置 (破坏自回归平滑性，易触发模板坍塌)

[Trial 9 设计：隐层连续软记忆反哺 (升级版 PCAM)]
dec_hidden ──> PCAM(Q=dec, K/V=MCP_Prototypes, Temp=1.5) ──> Gated Residual ──> Generator ──> Final Logits
                             ↑
                     连续高维隐层反哺 (无词表概率畸变，保留自回归流畅度)
```

### 2. 梯度流传导机理（反向对冲审计）

- **不可导 Buffer 的物理限制**：
  当前架构下，动量质心原型 `self.prototypes` 注册为 `register_buffer`，属于非叶子、不可导张量。其数值仅由 `_update_mcp_prototypes` 在 `with torch.no_grad():` 下根据样本重心通过 EMA 更新。因此，语言建模损失（NLL）的梯度**无法直接写入 MCP 原型张量**。
- **实际闭环反哺链路**：
  生成损失（NLL）通过 PCAM 的跨注意力机制，反向传导至：
  1. 解码器输出隐状态 $H_{\text{dec}}$ 及其上游 Transformer 解码层；
  2. PCAM 内部投影权重 $W_q, W_k, W_v, W_o$ 以及门控参数 $W_g, \gamma$；
  3. 通过编码器-解码器跨注意力回传至对话上下文编码器。
- **机理效果**：
  生成任务迫使解码器优化对这 32 个几何质心的检索方式，同时反向牵引上下文编码器产生更易被质心增强的隐状态，促使特征点云向对应质心靠拢，间接提升后续批次 EMA 动量质心的聚集质量。

---

## 四、三大工程决策项

### 决策 1：记忆注入层级（选择方案 A 适配器）
- **代码事实**：[`model.py:1257`](file:///e:/github/CASE/src/models/CASE/model.py#L1257) 中，解码层数由 `config.hop` 控制，默认 `hop=1`。
- **等价性论证**：在单层解码器下，方案 B（内部拼接 K-V 缓存）与方案 A（顶层输出跨注意力适配器）的数学拟合容量等价。
- **工程裁定**：方案 B 需要侵入修改 `DecoderLayer` 并重构 [`src/utils/decode/case.py`](file:///e:/github/CASE/src/utils/decode/case.py) 的单步生成逻辑，改动风险高；方案 A 在代码库中已有完整的 `apply_pcam` 接口埋点（覆盖训练、Greedy、Sampling、Beam 4 处入口）。因此**采用方案 A**。

### 决策 2：原型寻址与门控机制（平滑温度 + 门控偏置下限）
- **否决方案**：否决分类后验概率调制（$\hat{y}$ 准确率前中期仅 25%~35%，会造成错误级联）与离散 Top-K 掩码（不可导截断）。
- **确认方案**：
  1. **注意力温度平滑**：在点积打分阶段除以平滑温度 $\tau_{\text{attn}} = 1.5$：
     $$A = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k} \cdot \tau_{\text{attn}}}\right)$$
     提高注意力权重的信息熵，防止退化为近似 One-Hot 导致输出分布过锐。
  2. **门控偏置下限**：保留门控线性层初始化偏置 $b_{\text{gate}} = -1.0$，使初始 Sigmoid 门控值受限于 $\approx 0.269$，限制原型信息的强行渗漏。

### 决策 3：冷启动防护机制（$\tanh(\gamma)$，初始化为 0.0）
- **初值规范**：将可学习标量参数 $\gamma$ 显式初始化为 $0.0$：
  $$\tanh(0.0) = 0.0$$
- **防护机理**：在训练前期（Step 0 ~ 2000）原型尚未稳定收敛至高对齐度前，残差贡献在数值上为零，解码器完全依赖预训练词向量与常识上下文收敛，避免在原型未成熟阶段扰动语言模型基座。

---

## 五、新版 PCAM 算子形式化定义

新版 `PrototypeConditionedAttentionModule` 计算流如下：

1. **输入张量**：
   - 解码器输出隐状态：$H_{\text{dec}} \in \mathbb{R}^{B \times L \times D}$
   - 动量样本中心原型矩阵：$\mathcal{P} \in \mathbb{R}^{M \times D}$（$M=32$）
2. **多头平滑跨注意力**：
   $$Q = H_{\text{dec}} W_q, \quad K = \mathcal{P} W_k, \quad V = \mathcal{P} W_v$$
   $$A = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k} \cdot \tau_{\text{attn}}}\right), \quad \tau_{\text{attn}} = 1.5$$
   $$H_{\text{proto}} = \operatorname{Dropout}\left((A V) W_o\right)$$
3. **门控残差与平滑冷启动**：
   $$G = \sigma\left([H_{\text{dec}} \,\|\, H_{\text{proto}}] W_g + b_g\right), \quad b_g \sim \operatorname{Init}(-1.0)$$
   $$H_{\text{enhanced}} = \operatorname{LayerNorm}\left(H_{\text{dec}} + \tanh(\gamma) \cdot (G \odot H_{\text{proto}})\right), \quad \gamma \sim \operatorname{Init}(0.0)$$

---

## 六、代码修改清单（供执行者逐步操作）

### 修改项 1：注册 `--pcam_tau_attn` 到配置项
- **目标文件**：[`src/utils/config.py`](file:///e:/github/CASE/src/utils/config.py)
- **位置**：在 L153（`--pcam_gate_bias` 参数定义后）插入：
```python
    parser.add_argument("--pcam_tau_attn", type=float, default=1.5,
                        help="PCAM 交叉注意力的平滑温度系数 (默认 1.5，防止注意力权重过度锐化)")
```

---

### 修改项 2：升级 `PrototypeConditionedAttentionModule` 类
- **目标文件**：[`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py)
- **位置**：L506~L580
- **修改内容**：
```python
class PrototypeConditionedAttentionModule(nn.Module):
    """
    原型交叉记忆注意力模块 (PCAM V9 升级版):
    - 引入平滑温度系数 tau_attn (默认 1.5)，抑制条件概率分布过度锐化;
    - 引入可学习标量 gamma (初始化为 0.0)，通过 tanh(gamma) 实现严格的冷启动残差保护。
    """
    def __init__(self, d_model=300, num_heads=2, dropout=0.1, gate_bias=-1.0, tau_attn=1.5):
        super(PrototypeConditionedAttentionModule, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        self.head_dim = d_model // num_heads
        self.tau_attn = max(float(tau_attn), 0.1)
        self.scaling = (self.head_dim ** -0.5) / self.tau_attn

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.gate_linear = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.gate_linear.bias, gate_bias)
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.1)

        # 显式冷启动平滑残差标量，初始化为 0.0 (tanh(0.0) = 0.0)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.layer_norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, dec_output, prototypes):
        bsz, seq_len, _ = dec_output.size()

        # 1. 计算 Query
        q = self.q_proj(dec_output) * self.scaling
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 计算 Key, Value (自适应 2D 原型 [M, D])
        if prototypes.dim() == 2:
            k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
            v = self.v_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        else:
            _, k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 4. 计算原型上下文表征
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_proto = self.out_dropout(self.out_proj(context))

        # 5. 通道自适应门控与冷启动残差融合
        gate_input = torch.cat([dec_output, h_proto], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        enhanced_output = self.layer_norm(dec_output + torch.tanh(self.gamma) * (gate * h_proto))
        return enhanced_output
```

- **位置**：L1266~L1273（初始化传参同步更新）：
```python
        if self.use_pcam:
            self.pcam = PrototypeConditionedAttentionModule(
                d_model=config.hidden_dim,
                num_heads=getattr(config, 'pcam_heads', 2),
                dropout=getattr(config, 'pcam_dropout', 0.1),
                gate_bias=getattr(config, 'pcam_gate_bias', -1.0),
                tau_attn=getattr(config, 'pcam_tau_attn', 1.5)
            )
```

---

### 修改项 3：启动脚本 `main.sh` 传参解绑（规避 `store_true` 陷阱）

> [!CAUTION]
> **参数传值陷阱**：`--use_pcam`、`--use_emo_bias`、`--use_sparse_emo_bias` 在 `config.py` 中为 `action="store_true"`。
> 严禁写成 `--use_pcam True`（会报 `unrecognized argument: True`）或 `--use_emo_bias False`（会被强行赋值为 `True` 并报错）。

- **目标文件**：[`main.sh`](file:///e:/github/CASE/main.sh)
- **修改操作**：
  1. `EXP_NAME` 更新为 `v9_trial9`；
  2. 显式追加 PCAM 开关：
     ```bash
       --use_pcam \
       --pcam_heads 2 \
       --pcam_dropout 0.1 \
       --pcam_gate_bias -1.0 \
       --pcam_tau_attn 1.5 \
     ```
  3. **彻底删除**以下两行（不再传入任何偏置 flag）：
     ```bash
       # 彻底移除以下两行参数:
       # --use_sparse_emo_bias \
       # --emo_vocab_topk_ratio 0.15 \
     ```

---

## 七、验证与验收计划

### 1. 单元回归测试
在本地执行：
```powershell
& "E:\Dev\condaData\envs_dirs\cem_env\python.exe" -m unittest tests/test_sanity.py
```
需确认：
- 现有 16 项机制测试通过；
- PCAM 初始化与前向维度输出正确；
- 在 $\gamma=0.0$ 时，残差分支前向数值与基准严格对齐。

### 2. 评测指标验收阈值

| 指标 | 物理逻辑 | 验收合格线 | 突破线 |
| :--- | :--- | :---: | :---: |
| **Test PPL** | 原型先验降低解码不确定度，收复历史低位 | $\le 33.00$ | **$\le 32.70$**（追平 V6 T3 历史纪录 32.76） |
| **Test EMO_acc** | 原型受生成损失协同牵引，向心性增强 | $\ge 41.80\%$ | **$\ge 42.50\%$** |
| **Beam Dist-2** | **核心防御指标**：平滑温度 $\tau_{\text{attn}}=1.5$ 抑制概率过度锐化 | **$\ge 9.0\%$**（远超 V6 塌陷值 3.11%） | $\ge 11.0\%$ |
| **Greedy Dist-2** | 摆脱词表固定偏置，遣词造句恢复多样性 | $\ge 16.0\%$ | $\ge 17.5\%$ |
| **Alignment** | 原型参与解码任务优化，几何质心更贴合真实语义 | $\ge 0.935$ | $\ge 0.940$ |
