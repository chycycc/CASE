# CASE-EPCL V8 实验与文档导航手册

> **项目标识**：CASE-EPCL (V8 动量质心原型 MCP、加性硬边际 Arc-EPCL 与词表偏置退火优化)  
> **计算环境**：本地单卡 NVIDIA GeForce RTX 3050 Ti Laptop (4GB 显存, FP16 混合精度, Batch=8)  
> **当前状态**：全周期收官已结项。终局推荐旗舰为 **V8.2 F**（Step 39,999 黄金权重，Test PPL **32.83** / Greedy Dist-2 **16.79%** / Alignment **0.9514**）。后续已跨越至 V9 云端 24GB 大批次生产体系。

---

## 一、文档体系结构

`docs/v8/` 目录按功能分为三级子目录与根目录核心成果文档（共 11 篇 Markdown，消除碎片化）：

```text
docs/v8/
├── README.md                                         # [本文件] 全局导航看板与终局状态快照
├── v8_final_experiment_summary.md                    # V8 全生命周期终局实验总结与学术成果报告（精炼权威总结）
├── v8_experiment_log.md                              # V8 实验全周期统一大账（逐试验详细参数、机制与指标）
├── plans/                                            # 顶层架构指导与战略演进设计（共 2 篇）
│   ├── v8_momentum_centroid_prototypes_guidance.md   # V8 顶层架构指导（MCP 质心原型与 Arc-EPCL 数学假说）
│   └── v8.2_evolution_analysis_and_design.md         # V2 ~ V8.1 演进复盘与 V8.2 破局顶层设计
├── reports/                                          # 各版本闭环迭代复盘与流形审计报告（共 8 篇）
│   ├── v8.1_dual_track_iteration_report.md           # V8.1 双轨设计与实战闭环复盘（偏置注入 vs 软退火）
│   ├── v8.2_iteration_report.md                      # V8.2 初版破局闭环复盘（Top 15% 稀疏掩码与 PCGrad）
│   ├── v8.2_b_iteration_report.md                    # V8.2 B 破局迭代复盘（解绑退火与早停 patience 修复）
│   ├── v8.2_c_iteration_report.md                    # V8.2 C 全达标复盘（对称双向 PCGrad 与偏置余弦退火）
│   ├── v8.2_d_iteration_report.md                    # V8.2 D 终局复盘（动态情感加权与早产截断归因）
│   ├── v8.2_e_iteration_report.md                    # V8.2 E 全量达标复盘（统一锚点复位与退火保护期）
│   ├── v8.2_f_experiment_report.md                   # V8.2 F 全周期终局推荐旗舰实战复盘
│   └── v8.2_f2_manifold_audit_and_batch_upgrade_analysis.md # V8.2 F2 流形审计与批次升级动因分析
├── raw/                                              # 原始过程日志与打点流水账
│   └── v8_validation_curves.md                       # 各试验每 2,000 步的验证集打点数据归档 (170行)
└── images/                                           # 原型与隐空间流形 t-SNE 高清投影图谱
    ├── v8_trial1_manifold.png ~ v8_trial4_manifold.png
    └── v8_2_manifold.png ~ v8_2_f_manifold.png
```

---

## 二、文档职责与快速阅读指引

### 1. 终局总结与全景大账（必读入口）
- [v8_final_experiment_summary.md](file:///e:/github/CASE/docs/v8/v8_final_experiment_summary.md)  
  高度浓缩的 V8 收官报告，包含技术演化路线图（4 个阶段演进 Mermaid 流程）、核心学术指标统一大矩阵、帕累托前沿权衡分析与官方终局旗舰结论。
- [v8_experiment_log.md](file:///e:/github/CASE/docs/v8/v8_experiment_log.md)  
  全周期统一总账，详细记录 V8 Trial 1 至 Trial 4 以及后续各衍生版本的完整超参矩阵、验证集关键节点演变、测试集全量指标与流形图谱。

### 2. 顶层规划目录 (`plans/`)
- [v8_momentum_centroid_prototypes_guidance.md](file:///e:/github/CASE/docs/v8/plans/v8_momentum_centroid_prototypes_guidance.md)  
  V8 立项顶层设计，推导原型质心 EMA 滚动更新机制（$\mu=0.99$）及加性角度硬边际（$m=0.30$）。
- [v8.2_evolution_analysis_and_design.md](file:///e:/github/CASE/docs/v8/plans/v8.2_evolution_analysis_and_design.md)  
  复盘 V2 至 V8.1 的指标瓶颈，提出稀疏词表偏置（Top 15% Zero-Mask）与冲突梯度投影（PCGrad）设计。

### 3. 闭环迭代报告目录 (`reports/`)
- 各版本闭环报告均整合了“改动动机与代码实现”及“实测学术指标与因果归因”，彻底消除旧版中 Plan 与 Report 割裂复读的现象。
- 重点推荐：
  - [v8.2_c_iteration_report.md](file:///e:/github/CASE/docs/v8/reports/v8.2_c_iteration_report.md)：对称双向 PCGrad 与偏置余弦退火的经典达标范本；
  - [v8.2_f_experiment_report.md](file:///e:/github/CASE/docs/v8/reports/v8.2_f_experiment_report.md)：V8 全周期终局旗舰模型；
  - [v8.2_f2_manifold_audit_and_batch_upgrade_analysis.md](file:///e:/github/CASE/docs/v8/reports/v8.2_f2_manifold_audit_and_batch_upgrade_analysis.md)：定位单步批次 8 导致类簇重叠的物理瓶颈，直接推导向 V9 24GB 演进的必要性。


### 4. 原始打点数据 (`raw/`)
- [v8_validation_curves.md](file:///e:/github/CASE/docs/v8/raw/v8_validation_curves.md)  
  过程打点数据流水账，保留各实验每 2,000 步的 Valid PPL、EMO_acc、EMO_loss 原始观测值，供曲线拟合与历史稽核备查。

---

## 三、历代版本核心学术指标统一总览表 (测试集 5,255 样本)

| 架构版本 | 核心机制与实验变量 | 验证最低 PPL | 测试 PPL ↓ | EMO_acc ↑ | EMO_loss ↓ | Greedy Dist-2 ↑ | Sampling Unique ↑ | 原型对齐度 (Alignment) ↑ | 类簇紧实度 (DBI) ↓ | 状态定位 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CASE 原版** (ACL'23) | 双图编码 + MIM 局部互信息 | — | 35.37 | 40.20% | 2.2553 | 4.01% | 25.40% | — (无原型) | 5.6536 | 原始基线 |
| **V8 Trial 1** | 单变量替换: MCP 动量质心原型 ($\mu=0.99$) | 37.37 | 33.31 | 40.08% | 2.3407 | 42.52% | 100.0% | **0.9609** | — | 质心锚定突破 |
| **V8 Trial 2** | 统一挂载 `emotion_enc` + Arc-EPCL ($m=0.30$) | **36.58** | **32.62** | 39.39% | 2.4158 | — | 100.0% | 0.8845 | 4.7803 | PPL 极值突破 |
| **V8 Trial 4** | 双锚点解耦 + 时序软退火 | 37.38 | 33.40 | 39.66% | 2.8137 | — | 100.0% | 0.9393 | 5.4046 | 序列解耦探索 |
| **V8.1 Trial A** | 全词表偏置 + 全程联合微调 | 38.68 | 34.26 | 39.70% | 2.2354 | 16.20% | 97.40% | 0.9093 | — | 多样性引入 |
| **V8.2 初版** | Top 15% 稀疏偏置 + PCGrad (未退火早停) | 38.84 | 34.91 | 40.34% | **2.1059** | 12.98% | 97.80% | 0.9321 | 4.9882 | 稀疏掩码探索 |
| **V8.2 B** | 解绑余弦退火至 1e-5 + patience=12 | **37.13** | **33.03** | 40.38% | 2.5201 | **16.96%** | 97.40% | 0.9490 | — | 退火周期修复 |
| **V8.2 C** | 对称双向 PCGrad + 偏置时序余弦退火 | **37.09** | **32.84** | 41.16% | **2.2999** | 16.41% | 99.00% | 0.9498 | 5.1669 | 梯度冲突压制 |
| **V8.2 D** | 动态情感加权 + 帕累托早停 ($\alpha=8.0$) | 38.51 | 34.63 | **41.56%** | **2.0764** | 11.49% | 98.80% | 0.9387 | 4.9505 | 探索分类上限 |
| **V8.2 E** | 统一锚点复位 + 退火对齐 (24k) + 保护期 | **37.11** | **32.85** | 40.67% | **2.3336** | 16.14% | 99.00% | 0.9498 | 5.1799 | 保护期验证 |
| **V8.2 F** ⭐ | **钟形余弦加权 + 偏置下限 0.10 + 走完退火 40k** | **37.09** | **32.83** | 40.97% | 2.3458 | **16.79%** | 98.60% | **0.9514** | 5.0867 | **全周期终局推荐旗舰** |
| **V8.2 F2** | 强行提升 $\alpha=7.0$ 偏向分类 (牺牲 1.05 点 PPL) | 38.19 | 33.88 | **41.29%** | **2.1227** | 16.71% | **99.60%** | 0.9189 | **4.9414** | 权衡消融对照组 |
