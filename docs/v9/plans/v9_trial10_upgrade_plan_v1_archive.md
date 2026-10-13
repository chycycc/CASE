# [存档] CASE-EPCL V9 Trial 10 架构升级实施方案 v1 原文

> **文件性质**：v1 方案原文存档（规范化恒等 PCAM 与双流判别头 DPFC）。
> **存档原因**：v1 已被阶段 0 实测部分否决，现行为 [`v9_trial10_upgrade_plan.md`](v9_trial10_upgrade_plan.md)（v2 修订版，逐条标注保留/修正/失效）。
> **存档价值**：v1 中多条内容经阶段 0 检验后仍然成立，并被 v2 直接继承（见 v2 第零章状态对照表）；本文件为逐字原文，禁止修改，仅供追溯与引用。
> **原文范围**：自本行下方的分隔线起，为 v1 全文（原 344 行）。

---

# CASE-EPCL V9 Trial 10 架构升级实施方案：规范化恒等 PCAM 与双流判别头（DPFC）

> **文档性质**：Trial 10 生产级工程实施与技术规范（供后续 AI 工程师直接严格执行）。  
> **前置依赖**：Trial 9 取得全周期突破（Test PPL 32.79，Beam Dist-2 14.32%，Alignment 0.9415）；Step 6999 纯 PPL 最优检查点评测指令就绪。  
> **核心任务**：  
> 1. 重构 PCAM 适配器为规范化无损恒等架构（LayerNorm 移入增量分支，基座隐状态完全直通）；  
> 2. 落地双流原型-特征判别头（Dual-Stream Prototype-Feature Classifier, DPFC），激活 0.9415 原型判别能力；  
> 3. 锁定训练调度（继承 Trial 9 大批次高步频底座，严禁暴力拉高后程分类权重，坚决捍卫 Test PPL $\le 32.79$ 第一红线）。  
> **代码分支**：`v9-clean`  
> **算力环境**：AutoDL 单卡 NVIDIA GeForce RTX 4090D (24GB 显存, CUDA 12.1, PyTorch 2.1.0, 原生 FP32)

---

## 一、背景与前置状态

### 1. 基准现状（Trial 9 奠定的历史高位）
Trial 9 成功确立了单步物理批次 32、零累积（`ACCUM_STEPS=1`，有效更新步频 10,000 次）与升级版 PCAM 软记忆反哺的有效性：
- **Test PPL**：达到 **32.79**（精确值 32.7836），打破全项目历史纪录（V8.2 F 为 32.83，原版为 35.37）；
- **Beam (k=5) Dist-2**：达到 **14.32%**（创全项目历史新高，彻底破除 3.11% 坍塌阴影）；
- **Greedy Dist-2**：达到 **17.83%**（创全项目历史新高）；
- **原型对齐度 (Alignment)**：稳定在 **0.9415** 高位。

### 2. 当前核心痛点
在维持生成语言模型卓越质量的同时，系统当前存在两处明显的局部瓶颈：
1. **情感分类后程表征滞后**：测试集 EMO_acc 仅为 41.20%（较 Trial 7 的 42.25% 落后约 1.05pp），EMO_loss 反弹至 2.3310；
2. **PCAM 结构非严格恒等**：原代码外层包裹的 LayerNorm 导致冷启动 $\gamma=0$ 时输出为 $\text{LayerNorm}(h_t)$，微幅形变了基座解码器的输出尺度。

---

## 二、候选机制论证与排除记录 (严禁事项)

在方案设计与技术审查阶段，对以下两个潜在改进方向进行了严谨物理核验，**均予以坚决否决并列为实施红线**：

### 1. 否决方向 1：PCAM 引入对话级全局情感预测硬先验（严格禁止）
- **否决依据**：当前测试集情感分类准确率仅为 41.20%，全集近 **58.8%** 的样本全局情感判断是错误的。
- **级联失效风险**：若强行将全局情感预测分布作为偏置或加权注入 PCAM，近 60% 样本的词生成将被迫向错误的原型检索，严重诱发认知扭曲，必将直接摧毁 Trial 9 取得的 Beam Dist-2 历史突破（14.32%）。
- **代码物理事实**：PCAM 当前采用的是 **300 维通道级向量门控（Channel-wise Vector Gate）**（`nn.Linear(2*d_model, d_model)`，初始偏置 $-1.0$），本身已具备在语法通道抑制无情感词、在语义通道放行原型的解耦能力，无需且严禁引入全局有偏硬约束。

### 2. 否决方向 2：训练后半程暴力拉高分类损失保底权重（严格禁止）
- **否决依据**：项目历史对照组 **V8.2 F2**（查阅 [`docs/v8/v8_final_experiment_summary.md:L59`](file:///e:/github/CASE/docs/v8/v8_final_experiment_summary.md#L59)）强行在训练后程维持高分类权重，EMO_loss 虽降至 2.1227，但代价是 **Test PPL 从 32.83 恶化退化至 33.88（严重劣化 1.05 点）**！
- **多任务撕裂机理**：训练后半程学习率已降至 $10^{-5}$ 数量级，自回归 NLL 正在收敛至平坦极小值。强行增大辅助任务权重将直接撕裂共享底层编码器，破坏语言流形。**“Test PPL $\le 32.79$ 为第一核心红线”，严禁采用任何牺牲 PPL 的暴力加权手段**。

---

## 三、Trial 10 核心机制设计与形式化推导

### 1. 机制 1：规范化恒等 PCAM 适配器（Identity-Preserving PCAM Adapter）

```
[Trial 9 历史实现 (外层 LayerNorm 破坏恒等性)]
enhanced_output = LayerNorm(dec_output + tanh(γ) * (gate * h_proto))
                  ↑ 当 γ=0 时，输出为 LayerNorm(dec_output) ≠ dec_output

[Trial 10 规范化实现 (标准 Adapter 范式)]
h_proto_norm = LayerNorm(h_proto)
enhanced_output = dec_output + tanh(γ) * (gate * h_proto_norm)
                  ↑ 当 γ=0 时，输出严格等于 dec_output (无损恒等直通)
```

#### 数值量纲与尺度方差分析：
- **基座解码器输出** $h_t$：采用 Pre-LN 结构，各维度激活值方差天然稳定在 $\approx 1.0$；
- **增量分支内部** $\text{LayerNorm}(h_{\text{proto}})$：输出方差严格归一化为 $1.0$；
- **门控与冷启动**：$\text{gate} \in (0, 1)$，初始标量 $\tanh(\gamma) \to 0$；
- **结论**：增量分支与基座主干处于完全相同的数值量纲区间，不存在移出外层 LayerNorm 导致张量模长发散的风险，数学稳定性完全成立。

---

### 2. 机制 2：双流原型-特征判别头（Dual-Stream Prototype-Feature Classifier, DPFC）

#### （1）几何空间投影对齐
动量样本中心原型（MCP）$\mathbf{P} \in \mathbb{R}^{32 \times 300}$ 维护在 `projection_head` 之后的超球面流形上。计算原型预测分布时，必须显式通过投影头：
$$z_{\text{proj}} = \operatorname{Normalize}\left(\operatorname{projection\_head}(h_{\text{emo}})\right) \in \mathbb{R}^{B \times 300}$$
$$\text{logits}_{\text{proto}} = \frac{z_{\text{proj}} \cdot \mathbf{P}^\top}{\tau_{\text{proto}}}$$

#### （2）后验概率凸组合（防范置信度稀释与交叉熵对冲）
为规避原始 Logits 幅度不匹配引发的数值淹没，采用后验概率凸组合：
$$P_{\text{linear}} = \operatorname{Softmax}(\text{logits}_{\text{linear}})$$
$$P_{\text{proto}} = \operatorname{Softmax}(\text{logits}_{\text{proto}})$$
$$P_{\text{final}} = (1 - \alpha_{\text{dpfc}}) P_{\text{linear}} + \alpha_{\text{dpfc}} P_{\text{proto}}$$

#### （3）工程搜索区间与判定线
为防止过度稀释高置信度正确样本导致交叉熵对冲，超参网格搜索区间严格锁定在：
$$\alpha_{\text{dpfc}} \in [0.05, 0.25], \quad \tau_{\text{proto}} \in [0.05, 0.15]$$
**黄金判定准则**：以“`EMO_acc` 提升 $\ge 0.8\text{pp}$ 且 `EMO_loss` 不恶化”为锁定标准。

---

## 四、三大关键工程决策项

### 决策 1：PCAM 归一化重构必须从头训练（Scratch）
- **技术事实**：修改 LayerNorm 位置改变了自回归生成的前向计算图与隐状态方差分布；
- **执行纪律**：**严禁直接加载 Trial 9 旧权重进行推断测试**。必须从初始随机种子出发，在单步 32 零累加底座上完整训练 10,000 iter。

### 决策 2：DPFC 架构解耦与自适应启用
- **解耦设计**：DPFC 逻辑作为独立开关 `--use_dpfc` 实现于模型中。
- **推断与评测兼容**：默认在前向计算 `pred_emotion` 时生效，不改变损失反向传播图（纯推断期后验融合）；亦可支持在验证集上通过离线脚本免训完成超参扫描。

### 决策 3：训练损失调度严格维持 Trial 9 最优解
- **调度配置**：
  - 学习率：LR=$5.76 \times 10^{-4}$，Cosine 退火（3000 $\to$ 9000 步，最低 $1.73 \times 10^{-5}$）；
  - 情感加权：维持原版钟形加权（2500 $\to$ 5000 步，峰值 1.25 后回落至 1.0）；
  - 绝对不尝试暴力保底加权，彻底杜绝 PPL 恶化风险。

---

## 五、算子形式化数学定义

### 1. 规范化 PCAM 算子
输入解码器输出 $H_{\text{dec}} \in \mathbb{R}^{B \times L \times D}$ 与动量原型 $\mathcal{P} \in \mathbb{R}^{32 \times D}$：
1. **多头平滑跨注意力**：
   $$Q = H_{\text{dec}} W_q \cdot \frac{1}{\sqrt{d_k} \cdot \tau_{\text{attn}}}, \quad K = \mathcal{P} W_k, \quad V = \mathcal{P} W_v \quad (\tau_{\text{attn}}=1.5)$$
   $$A = \operatorname{Softmax}\left(Q K^\top\right)$$
   $$H_{\text{proto}} = \operatorname{Dropout}\left((A V) W_o\right)$$
2. **分支内归一化与恒等残差注入**：
   $$\widetilde{H}_{\text{proto}} = \operatorname{LayerNorm}(H_{\text{proto}})$$
   $$G = \sigma\left([H_{\text{dec}} \,\|\, \widetilde{H}_{\text{proto}}] W_g + b_g\right), \quad b_g \sim \operatorname{Init}(-1.0)$$
   $$H_{\text{enhanced}} = H_{\text{dec}} + \tanh(\gamma) \cdot \left(G \odot \widetilde{H}_{\text{proto}}\right), \quad \gamma \sim \operatorname{Init}(0.0)$$

---

## 六、详细代码修改清单（供执行者逐步操作）

### 修改项 1：注册 DPFC 相关参数至 `src/utils/config.py`
- **目标文件**：[`src/utils/config.py`](file:///e:/github/CASE/src/utils/config.py)
- **位置**：在 `--use_pcam` 对应参数组后（约 L155 之后）插入：
```python
    # V9 Trial 10: 双流原型-特征判别头 (DPFC)
    parser.add_argument("--use_dpfc", action="store_true", default=False,
                        help="是否启用双流原型-特征判别头 (DPFC) 进行后验概率融合")
    parser.add_argument("--dpfc_alpha", type=float, default=0.15,
                        help="DPFC 原型预测分支凸组合权重 (默认 0.15)")
    parser.add_argument("--dpfc_tau", type=float, default=0.10,
                        help="DPFC 原型余弦相似度 Softmax 温度系数 (默认 0.10)")
```

---

### 修改项 2：PCAM 规范化无损恒等重构
- **目标文件**：[`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py)
- **位置**：`PrototypeConditionedAttentionModule.forward`（约 L576~L585）
- **原代码**：
```python
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
- **替换为**：
```python
        # 4. 计算原型上下文表征并实施增量分支内归一化
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_proto = self.out_dropout(self.out_proj(context))
        h_proto_norm = self.layer_norm(h_proto)

        # 5. 通道自适应门控与标准无损恒等残差注入 (gamma=0 时严格恒等直通)
        gate_input = torch.cat([dec_output, h_proto_norm], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        enhanced_output = dec_output + torch.tanh(self.gamma) * (gate * h_proto_norm)
        return enhanced_output
```

---

### 修改项 3：模型新增 DPFC 概率融合推理方法
- **目标文件**：[`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py)
- **位置**：在 `CASE` 模型类中新增方法 `get_dpfc_emotion_probs`：
```python
    def get_dpfc_emotion_probs(self, target_emotion_feat, emotion_logits, alpha=0.15, tau=0.10):
        """
        [V9 Trial 10: DPFC] 双流原型-特征后验概率融合
        target_emotion_feat: [B, 300] 进入 EPCL 投影头的目标特征 (默认 fine_emotion)
        emotion_logits: [B, num_classes] 线性分类头输出
        返回: final_probs [B, num_classes] 融合后的后验概率分布
        """
        p_linear = F.softmax(emotion_logits, dim=-1)
        if not getattr(self, 'use_dpfc', False) or not hasattr(self, 'epcl_criterion'):
            return p_linear

        with torch.no_grad():
            # 必须通过 projection_head 映射至超球面流形 (与 EPCL 动量原型空间严格对齐)
            proj = self.epcl_criterion.projection_head(target_emotion_feat)
            z_proj = F.normalize(proj, p=2, dim=-1)
            protos = F.normalize(self.epcl_criterion.current_prototypes, p=2, dim=-1)
            proto_logits = torch.matmul(z_proj, protos.T) / max(tau, 0.01)
            p_proto = F.softmax(proto_logits, dim=-1)

        p_final = (1.0 - alpha) * p_linear + alpha * p_proto
        return p_final
```

---

### 修改项 3.1：`train_one_batch` 挂载 DPFC 与评测期 EMO_loss 闭环
- **目标文件**：[`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py)
- **位置**：`CASE.train_one_batch` 情感预测代码块（约 L1993 原 `pred_emotion` 计算处）
- **核心实施防范纪律**：
  - **严禁误删上游调度**：原代码 L1951~L1992 包含多任务余弦退火、软冻结与**核心钟形余弦加权调度（Bell Ramp）**，必须完整保留，原封不动！
  - **精准重载变量**：`train_one_batch` 最终返回的损失元组中的第 8 个元素是 `emotion_loss.item()`（而非 `emotion_loss_raw`）。评测期必须直接重载 `emotion_loss`，方可确保评测指标真实反映 DPFC 收益。
- **修改代码（仅在 L1993 原处替换）**：
```python
            # [原代码 L1951~L1992 保持原封不动，计算出 emotion_logits 与训练期的 emotion_loss]
            # ... (保留 use_cosine_anneal, emo_head_active, use_emo_loss_ramp 等所有调度) ...

            # [V9 Trial 10: DPFC 仅在 L1993 原地替换 pred_emotion 计算与评测期损失重载]
            if getattr(self, 'use_dpfc', False):
                alpha_dpfc = getattr(config, 'dpfc_alpha', 0.15)
                tau_dpfc = getattr(config, 'dpfc_tau', 0.10)
                p_final = self.get_dpfc_emotion_probs(target_emotion_feat, emotion_logits, alpha=alpha_dpfc, tau=tau_dpfc)
                pred_emotion = np.argmax(p_final.detach().cpu().numpy(), axis=1)
                if not train:
                    # 评测期将最终返回的 emotion_loss 原地重载为融合后验概率的 NLL 损失
                    eps = 1e-12
                    emotion_loss = F.nll_loss(torch.log(p_final.clamp(min=eps)), batch["program_label"])
            else:
                pred_emotion = np.argmax(emotion_logits.detach().cpu().numpy(), axis=1)

            emotion_acc = accuracy_score(batch["program_label"].detach().cpu().numpy(), pred_emotion)
            self.current_pred_emotion = pred_emotion
```

---

### 修改项 4：全自动评测流水线参数注册与透传
- **目标文件**：[`src/scripts/eval_pipeline.py`](file:///e:/github/CASE/src/scripts/eval_pipeline.py)
- **位置 1**：在 `main()` 函数参数解析处（约 L629）注册 DPFC 命令行参数：
```python
    parser.add_argument("--use_dpfc", action="store_true", default=False, help="是否启用双流原型-特征判别头 (DPFC)")
    parser.add_argument("--dpfc_alpha", type=float, default=0.15, help="DPFC 原型预测分支凸组合权重 (默认 0.15)")
    parser.add_argument("--dpfc_tau", type=float, default=0.10, help="DPFC 原型余弦相似度 Softmax 温度 (默认 0.10)")
```
- **位置 2**：在 `load_v8_model` 函数（约 L210~L245）及模型实例化后，将参数对齐至模型配置：
```python
    config.use_dpfc = getattr(args, 'use_dpfc', False)
    config.dpfc_alpha = getattr(args, 'dpfc_alpha', 0.15)
    config.dpfc_tau = getattr(args, 'dpfc_tau', 0.10)
```

---

### 修改项 5：配置启动脚本 `main.sh` 为 Trial 10（含评测调用对齐）
- **目标文件**：[`main.sh`](file:///e:/github/CASE/main.sh)
- **修改点 1**：L14 默认实验代号变更为 `v9_trial10`；
- **修改点 2**：在主训练 Python 命令参数中添加：
  ```bash
  --use_dpfc \
  --dpfc_alpha 0.15 \
  --dpfc_tau 0.10 \
  ```
- **修改点 3**：在 L143 自动调用 `src/scripts/eval_pipeline.py` 处，同步传入 DPFC 评测参数：
  ```bash
  --use_dpfc \
  --dpfc_alpha 0.15 \
  --dpfc_tau 0.10 \
  ```
  杜绝评测脚本将命令行参数丢弃导致自动化指标大账被架空。

---

### 修改项 5.1：外围手动评测脚本 `eval.sh` 参数透传支持
- **目标文件**：[`eval.sh`](file:///e:/github/CASE/eval.sh)
- **位置**：L50~L56 调用 `src/scripts/eval_pipeline.py` 命令末尾添加 `"${@:5}"`：
```bash
${pythonpath} src/scripts/eval_pipeline.py \
  --model_path "${MODEL_PATH}" \
  --eval_target "${EVAL_TARGET}" \
  --results_path "results/v9/${OUTPUT_TAG}_eval.txt" \
  --plot_path "results/v9/${OUTPUT_TAG}_manifold.png" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${GPU_ID}" \
  "${@:5}"
```
- **收益**：支持手动调用脚本执行全量评测时透明透传 `--use_dpfc` 等任意扩展参数（例如 `./eval.sh save/v9_trial10 0 64 ppl --use_dpfc`）。

### 修改项 6：编写离线网格搜索验证脚本
- **新建目标文件**：`scratch/eval_dpfc_offline.py`
- **功能与细节规范**：读取 `CASE_7499_37.7145_composite_best` 权重，模型置为 `model.eval()`。从测试集提取 `target_emotion_feat`（即 `fine_emotion`）与 `emotion_logits`，运行二维网格搜索（$\alpha \in [0.05, 0.25], \tau \in [0.05, 0.15]$），输出不同参数下的混淆矩阵、EMO_acc 与 EMO_loss 变化曲线。

---

## 七、单测防护网与冒烟测试验证规程

在云端启动生产训练前，必须在本地执行如下 4 层防护验证：

1. **语法与静态编译检查**：
   ```bash
   python -m py_compile src/models/CASE/model.py src/utils/config.py src/scripts/eval_pipeline.py
   ```
2. **PCAM 恒等无损冷启动单测**：
   编写测试验证当 $\gamma=0.0$ 时，输入任意随机张量 $X$，PCAM 输出必须满足：
   ```python
   assert torch.allclose(enhanced_output, dec_output, atol=1e-6)
   ```
3. **核心数学与组件单元测试**：
   ```bash
   python tests/test_sanity.py
   ```
4. **端到端多解码分支真实 Batch 冒烟测试**：
   ```bash
   python tests/smoke_test_pcam_integration.py
   ```
   覆盖模型加载、单步前向/反向传播、Greedy 贪婪解码、Sampling 采样解码以及 BeamSearch 束搜索，确保无损恒等 PCAM 在所有解码循环中的张量维度兼容性。

---

## 八、AutoDL 生产运行指令与阶段验收标准

### 1. 云端启动指令
```bash
# 1. 本地代码与脚本通过 scp 增量同步至 AutoDL
# 2. 在 tmux 守护会话中启动生产训练:
./main.sh 24G v9_trial10
```

### 2. 阶段验收红线矩阵

| 指标维度 | Trial 9 历史基准 | **Trial 10 核心验收红线** | 预期冲刺目标 | 判定性质 |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | 32.79 | **$\le 32.79$ (第一不可退化红线)** | **32.65 ~ 32.75** | 核心生命线 |
| **Beam (k=5) Dist-2 ↑** | 14.32% | **$\ge 14.00\%$** | **$\ge 14.50\%$** | 多样性底线 |
| **Greedy Dist-2 ↑** | 17.83% | **$\ge 17.50\%$** | **$\ge 18.00\%$** | 连贯多样性 |
| **Test EMO_acc ↑** | 41.20% | **$\ge 42.00\%$** | **$\ge 42.50\%$** | DPFC 核心检验点 |
| **Test EMO_loss ↓** | 2.3310 | **$\le 2.2500$ (不恶化)** | **$\le 2.1500$** | 置信度稀释防线 |
| **Centroid Alignment ↑** | 0.9415 | **$\ge 0.9400$** | **$\ge 0.9450$** | 几何结构稳态 |

---

## 九、面向同行评议（审稿人）的方法论叙事规划

后续在论文方法与消融章节中，按如下规范建立完整创新叙事链条：
1. **命名定义**：双流原型-特征判别头（Dual-Stream Prototype-Feature Classifier, DPFC）；
2. **理论论述**：揭示传统单层线性分类器是局部判别头，极易在自回归生成尾程产生表征滞后；而 EPCL 的 EMA 动量质心表征全局几何流形。双流头将局部判别与全局流形几何距离在后验概率空间显式统一，是对比学习先验反哺下游判别的完整架构范式；
3. **消融矩阵**：主表配套消融实验：
   - `Baseline (CASE 原版)`
   - `+ EPCL (Trial 8)`
   - `+ PCAM (Trial 9)`
   - `+ DPFC (Trial 10 完整旗舰)`
   - `Ablation: Linear Head Only vs NPC Only vs DPFC Fusion`
