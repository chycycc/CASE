# CASE-EPCL 进阶创新与架构演进上下文 (Context Handoff)

## 1. 当前项目基线与核心痛点
本项目的目标是改进 ACL 2023 提出的 **CASE (Coarse-to-Fine Cognition and Affection)** 移情对话生成模型。

在 V3 阶段，我们成功将 **EPCL (Emotion Prototype Contrastive Learning, 全局原型级对比学习)** 移植到了 CASE 架构中，并实施了 **“双通道解耦 (Dual-dimensional Decoupling)”**，取得了如下突破：
- **Dist-2 多样性**：3.52% -> **4.06%**（突破原文献天花板 4.01%）
- **Emo Acc 准确率**：稳固在 **40.65%**（超越原文 40.20%）

**核心痛点与下一步创新动机**：
目前的创新仍然存在“生搬硬套”的局限。EPCL 原本是为缺乏对比机制的简单模型（如 CEM）设计的。而 CASE 模型本身不仅非常臃肿（COMET 常识图谱 + ConceptNet 概念图谱），而且内置了 **Mutual Information Maximization (MIM，实例级互信息最大化)**。
在此基础上叠加 EPCL 属于“同类机制堆叠”（局部对比 + 全局对比），虽然依靠架构解耦防住了特征坍缩并挤出了 0.05% 的提升，但在底层数学逻辑和架构深度上，缺乏针对 CASE 独有的“多模态/多级图谱网络”的深度适配与内生创新。

## 2. 推荐的探索方向
新的会话应重点解决“机制适配度”与“架构冗余”问题，建议探索以下方向：
1. **多模态图谱对比学习 (Graph-aware Contrastive Learning)**：与其在最终的融合特征上做 EPCL，不如针对 COMET 或 ConceptNet 的图注意力输出层做更细粒度的结构化对比。
2. **动态路由或稀疏门控 (Dynamic Routing / MoE)**：CASE 当前的 `emo_gate`过于死板，能否引入动态路由机制，让模型根据上下文自适应决定使用多少常识和多少情感？
3. **对比学习的互信息下界优化 (InfoNCE vs MIM)**：重构 CASE 的 MIM 损失，将其与 EPCL 的 InfoNCE 损失进行数学层面的统一，而不是简单相加。

## 3. 推荐挂载的 Skills 列表
在新会话中，强烈建议 @ 以下 Skills 以获得最专业的科研与架构辅助：

*   **`claude-scientific-skills`**：用于高阶的学术研究逻辑推演、数学公式论证和科研痛点分析。
*   **`deep-research`**：用于检索 2023-2026 年关于移情对话 (Empathetic Dialogues)、图神经网络对比学习 (Graph Contrastive Learning) 的最新 SOTA 论文。
*   **`hugging-face-papers`**：用于快速解析和提取深层学术论文的架构图和模型公式。
*   **`context-management-context-restore`**：用于在多会话之间恢复和管理复杂的架构知识。
*   **`python-pro`** / **`senior-architect`**：用于后续可能发生的重度模型代码（PyTorch）重构。

## 4. 新会话启动指南 (Prompt)
在新会话的第一次对话时，请直接发送以下指令：
> “你好，请读取 `docs/CASE_EPCL_Next_Phase_Context.md` 和 `docs/v3/v3_final_walkthrough.md` 以了解 CASE 项目的架构背景与 V3 实验成果。我们现在需要抛弃简单的 EPCL 堆叠，针对 CASE 的双图谱网络和 MIM 机制，设计一种深度融合的全新对比学习架构。请运用你的科研技能进行分析并给出 3 个具有顶会（ACL/EMNLP）创新潜力的架构改造方案。”
