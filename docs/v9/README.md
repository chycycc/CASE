# CASE-EPCL V9 实验与文档导航手册

> **项目标识**：CASE-EPCL (V9 生产级大批次与原型对比学习优化)  
> **计算环境**：AutoDL 单卡 NVIDIA GeForce RTX 4090D (24GB 显存, CUDA 12.1, PyTorch 2.1.0, FP32)  
> **当前状态**：Trial 8 深度复盘完成；Trial 9（PCAM 原型注意力调制重构）已完成代码重构与单测验收，待云端启动。

---

## 一、文档体系结构

`docs/v9/` 目录按功能分为三级子目录与全局总账文档：

```text
docs/v9/
├── README.md                                    # [本文件] 全局导航看板与状态快照
├── v9_experiment_log.md                         # 全周期统一学术大账（权威指标源与详细记录）
├── plans/                                       # 架构演进与阶段执行规划
│   ├── v9_architecture_roadmap.md               # V9 架构路线图与阶段战略规划
│   ├── v9_phase7_execution_plan.md              # Phase 7 架构优化、16 项工程修复与 Trial 7 执行计划
│   └── v9_pcam_upgrade_plan.md                  # PCAM 原型注意力调制模块架构升级方案
├── reports/                                     # 各试验深度报告与单测验收记录
│   ├── v9_trial7_report.md                      # Trial 7 实验复盘、对齐失效归因与决策树分析
│   ├── v9_trial8_report.md                      # Trial 8 实验复盘、强正则下表征崩塌与消融分析
│   └── v9_trial9_pcam_implementation_report.md  # Trial 9 PCAM 模块工程实现与冒烟测试报告
└── guides/                                      # 端云协同与运维标准操作规程 (SOP)
    └── v9_autodl_operation_and_sync_guide.md    # AutoDL 极速部署、后台持久化、产物 5 矩阵与 AI 闭环手册
```

---

## 二、文档导航与职责划分

### 1. 全局实验大账
- [v9_experiment_log.md](file:///e:/github/CASE/docs/v9/v9_experiment_log.md)  
  记录 V9 全周期所有 Trial（Trial 1 至 Trial 9）的核心超参配置、全量评测指标（Test PPL、EMO_acc、EMO_loss、Centroid Alignment、DBI、Dist-2、Unique）、流形可视化图表与物理机制剖析。为全项目唯一权威数据大账。

### 2. 方案规划目录 (`plans/`)
- [v9_architecture_roadmap.md](file:///e:/github/CASE/docs/v9/plans/v9_architecture_roadmap.md)  
  梳理从 4GB 本地环境向 24GB 云端大批次演进的系统拓扑，定义原型质心对齐与互信息最大化（MIM）的协同机制。
- [v9_phase7_execution_plan.md](file:///e:/github/CASE/docs/v9/plans/v9_phase7_execution_plan.md)  
  详细记录 P0-1 至 P0-6 共 16 项代码异味修复工程、单测防护网建立过程以及 Trial 7 调优方案。
- [v9_pcam_upgrade_plan.md](file:///e:/github/CASE/docs/v9/plans/v9_pcam_upgrade_plan.md)  
  推导并设计 PCAM（原型条件注意力模块）的软反哺架构，引入 $\tau_{\text{attn}}=1.5$ 锐化与 $\tanh(\gamma)$ 冷启动门控机制，替代原硬截断方式。

### 3. 实验报告目录 (`reports/`)
- [v9_trial7_report.md](file:///e:/github/CASE/docs/v9/reports/v9_trial7_report.md)  
  分析 Trial 7 中降低峰值学习率（$5.76\times 10^{-4}$）对情感分类准确率（42.25%）与 PPL（33.58）的影响，定位质心对齐度瓶颈。
- [v9_trial8_report.md](file:///e:/github/CASE/docs/v9/reports/v9_trial8_report.md)  
  剖析 Trial 8 物理批次 32 零累加策略下，Test PPL 刷新至 32.97 的机制，复盘旧版磁盘清理逻辑误删双检查点的问题，明确表征收缩与多样性平衡点。
- [v9_trial9_pcam_implementation_report.md](file:///e:/github/CASE/docs/v9/reports/v9_trial9_pcam_implementation_report.md)  
  详细记录 PCAM 算子在 `model.py`、`config.py`、`main.sh`、`eval_pipeline.py` 的集成修改，以及 16 项回归测试与单测验证结果。

### 4. 运行规范目录 (`guides/`)
- [v9_autodl_operation_and_sync_guide.md](file:///e:/github/CASE/docs/v9/guides/v9_autodl_operation_and_sync_guide.md)  
  涵盖本地精简打包（260MB）、云端学术加速直连下载 GloVe（30秒内）、tmux 后台守护、5 项核心产物标准化回传与本地 AI 4 步归档闭环。

---

## 三、历史 Trials 核心指标速查总表

| 实验代号 | 核心超参与配置变动 | 验证 PPL | 测试 PPL ↓ | 情感分类 Acc ↑ | 质心对齐度 ↑ | 类簇紧实度 (DBI) ↓ | Beam Dist-2 ↑ | 状态说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CASE ACL'23** | 原版基线 (4GB, B=8) | — | 35.37 | 40.20% | 0.0000 | 5.6536 | — | 原始基线 |
| **V8.2 F 旗舰** | 本地 4GB 混合精度收官版 | 37.0876 | **32.83** | 40.97% | **0.9514** | 4.7803 | — | 核心参照线 |
| **V9 Trial 2** | 24GB 首批次收敛版, 截获 Step 5k | 41.2357 | 35.63 | 40.30% | 待复测 | — | — | 基准建立 |
| **V9 Trial 3** | 余弦退火, $\alpha_{\text{uni}}=1.5, \lambda_{\text{epcl}}=0.07$ | 39.7587 | 34.34 | 40.65% | 0.9270 | — | — | 退火验证 |
| **V9 Trial 4** | Class-Aware Uniformity, $\lambda_{\text{epcl}}=0.03$ | 38.6506 | 33.19 | 40.82% | 0.9290 | — | — | 动态调度 |
| **V9 Trial 6** | 单变量 $\alpha_{\text{uni}}=0.8$, decay 3k→9k | 38.7629 | 33.62 | 41.88% | 0.9344 | — | — | 尺度收敛 |
| **V9 Trial 7** | 下调峰值 LR ($5.76\times 10^{-4}$), $\alpha=5.0$ 评分 | 38.4005 | 33.58 | **42.25%** | 0.9353 | — | — | 分类突破 |
| **V9 Trial 8** | 单步物理 B=32 (更新 10k 次), 双权重保存 | **37.8523** | **32.97** | 40.46% (PPL最优) / 43.35% (复合) | 0.9353 | — | 12.76% | 逼近 V8.2F |
| **V9 Trial 9** | PCAM 软反哺注意力, $\tau=1.5, \tanh(\gamma)$ | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | **就绪待跑** |

---

## 四、工程与运行约束

1. **分支管理**：所有代码、配置与文档变更均归集至 `v9-clean` 分支；
2. **权重隔离**：`save/` 目录下的大体积权重文件禁止纳入 Git 追踪，仅保留结构化元数据 `eval_metrics.json`；
3. **环境健康**：代码提交前必须执行 `python tests/test_sanity.py`，保证 16 项回归测试全部通过；
4. **注释与文档规范**：所有代码行内注释、Docstring 及工程文档均使用规范中文。
