# V3-P4: 双通道解耦对比学习 (Dual-dimensional Decoupled Contrastive Learning)

在 P3 实验中，我们证明了对单一特征通道施加过强的聚类约束（EPCL + CE）会导致生成多样性（Dist-2）崩溃。为了突破这个帕累托边界，我们必须在架构上将“准确率负责特征”与“多样性负责特征”进行物理隔离。

## 背景与问题诊断

在当前的 CASE 架构中，情感分类和文本生成共享了同一个融合特征：
```python
emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
emotion_logits = self.emotion_linear(self.emo_dropout(emotion_enc)) # 分类器
# emotion_enc 随后也被送入 Decoder
```
在这个结构下，交叉熵分类损失 (CE Loss) 的梯度会通过 `emotion_enc` 强行回传给 `concept_enc`（概念图谱特征）。这意味着，原本富含常识和语义多样性的 `concept_enc`，为了迎合分类任务，被迫收敛到了同质化的情感质心，导致模型词汇单一化。

## P4 核心解耦方案 (Decoupling)

我们将实施 **“分类-生成特征解耦 (Classifier-Generator Decoupling)”** 策略：

1. **准确率专职通道 (Accuracy Channel)**：
   - 情感分类器 (`emotion_linear`) **只接收** `fine_emotion`，完全不再接收 `concept_enc`。
   - `fine_emotion` 继续接受 EPCL 和 CE Loss 的双重聚类约束，确保 Emo Acc 的绝对稳定。
2. **多样性专职通道 (Diversity Channel)**：
   - `concept_enc` 将**彻底摆脱**分类交叉熵 (CE Loss) 的梯度更新。
   - 它将完全由解码器生成损失 (Decoder CE Loss)、BoW 损失和 KL 散度来驱动，从而最大程度保留知识图谱带来的发散性语义。
   - 融合后的 `emotion_enc`（由 `emo_gate` 混合）仅供应给 Decoder，不再流向分类器。

## Proposed Changes

### `src/models/CASE/model.py`

#### [MODIFY] `model.py`
在 `forward` 函数中，修改情感分类器的输入逻辑：

**修改前 (P3):**
```python
emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
emo_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion

# ... [EPCL loss calculation] ...

if self.dataset == "ED":
    emotion_logits = self.emotion_linear(self.emo_dropout(emotion_enc))
```

**修改后 (P4):**
```python
emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
emo_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
# 生成用融合特征 (仅送入 Decoder)
emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion

# ... [EPCL loss calculation] ...

# 【P4 核心解耦】分类器仅使用 fine_emotion，保护 concept_enc 不被分类梯度同质化
if self.dataset == "ED":
    emotion_logits = self.emotion_linear(self.emo_dropout(fine_emotion))
```

> [!NOTE]
> 仅仅一行代码的改变，我们在数学图谱上切断了 CE Loss 流向 `concept_enc` 的反向传播路径。

## 验证计划 (Verification Plan)

### 预期结果 (Hypothesis)
1. **Emo Acc**：由于 `fine_emotion` 单独承担分类任务，结合 EPCL 约束，预计维持在 40% 以上。
2. **Dist-2**：由于 `concept_enc` 被解放，纯粹为了生成多样性服务，Decoder 能够利用到更丰富的常识词汇，Dist-2 有望突破基线 3.52%，向 4.01% 逼近。
3. **PPL**：可能会有轻微回升（因为生成的词汇更丰富了，不再总是最安全的词），但保持在合理水平（< 39.0）。

### 执行命令
得到你的确认后，我将：
1. 修改 `model.py`。
2. 使用以下命令在后台启动 P4 训练：
```bash
conda run -n cem_env python main.py --dataset ED --woStrategy --batch_size 8 --pretrain --pretrain_epoch 4 --warmup 24000 --epcl_warmup 6000 --epcl_freeze_step 28000 --fine_weight 0.2 --coarse_weight 1.0 --seed 13 --gpu 0 --save_path save/epcl_v3_p4_decoupled > train_v3_p4.log 2>&1
```

## User Review Required
> [!IMPORTANT]
> 这种解耦方式非常巧妙且改动极小。你是否同意我们将分类器的输入从 `emotion_enc` 改为纯粹的 `fine_emotion`？确认后我将立即开始执行。
