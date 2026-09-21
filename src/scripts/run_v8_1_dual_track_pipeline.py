# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.1 双轨实验全自动化执行流水线 (Dual-Track SOP Runner)

流水线设计:
1. 阶段一 (Trial A): 
   - 启动 V8.1-Trial A: 词表层情感偏置注入 (KEMP Vocab Bias) + 全程联合微调 (disable_freeze)。
   - 训练完成自动执行评测流水线 (eval_v8_pipeline.py --use_emo_bias)。
   - 物理清理多余权重，保留唯一黄金检查点。
2. 阶段二 (Trial B):
   - 启动 V8.1-Trial B: 平滑余弦退火解耦 (Cosine Soft Annealing) + 复合早停机制 (Composite Early Stopping)。
   - 训练完成自动执行评测流水线 (eval_v8_pipeline.py)。
   - 物理清理多余权重，保留唯一黄金检查点。
3. 阶段三 (学术对比报告):
   - 汇总 Trial A 与 Trial B 的关键学术指标 (PPL, Distinct-1/2, EMO_acc, 聚类轮廓系数, 均匀度等)。
   - 输出 docs/v8/v8.1_dual_track_experiment_report.md。
"""

import os
import sys
import time
import shutil
import subprocess
import argparse

# 确保导入路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 针对 Windows 控制台强制配置 UTF-8 输出与错误替换
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def safe_print(msg):
    try:
        print(msg, flush=True)
    except Exception:
        try:
            # 兜底：若控制台无法输出某些 Unicode 字符，使用 gbk 替换模式
            clean_msg = str(msg).encode("gbk", errors="replace").decode("gbk")
            print(clean_msg, flush=True)
        except Exception:
            pass

PYTHON_EXE = sys.executable

def log_header(title):
    safe_print("\n" + "=" * 80)
    safe_print(f"[*] {title}")
    safe_print("=" * 80 + "\n")

def check_disk_space(min_gb=10.0):
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024 ** 3)
    safe_print(f"[Disk Audit] 项目所在磁盘剩余空间: {free_gb:.2f} GB (红线阈值: {min_gb:.2f} GB)")
    if free_gb < min_gb:
        raise RuntimeError(f"磁盘空间不足！当前可用 {free_gb:.2f} GB < 设定红线 {min_gb:.2f} GB，终止以防崩溃！")
    return free_gb

def run_command_with_logging(cmd, log_file_path, desc=""):
    log_header(f"开始执行任务: {desc}")
    safe_print(f"[CMD]: {' '.join(cmd)}")
    safe_print(f"[Log]: {log_file_path}\n")

    log_dir = os.path.dirname(os.path.abspath(log_file_path))
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with open(log_file_path, "w", encoding="utf-8", errors="replace") as f_log:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        
        last_print_time = time.time()
        for line in process.stdout:
            f_log.write(line)
            f_log.flush()
            
            # 定期向主控制台输出关键节点或定期心跳
            curr_time = time.time()
            if any(k in line for k in ["[Adaptive Freeze", "[*] Step", "[-] Step", "Epoch", "Val PPL", "loss_train", "results", "[+]"]):
                safe_print(f"[{desc}] {line.strip()}")
                last_print_time = curr_time
            elif curr_time - last_print_time > 60.0:
                safe_print(f"[{desc} 心跳] 正在平稳训练中... (最新行: {line.strip()[:60]}...)")
                last_print_time = curr_time

        ret_code = process.wait()
        if ret_code != 0:
            raise RuntimeError(f"任务 [{desc}] 异常退出，退出码: {ret_code}，请检查日志: {log_file_path}")
        safe_print(f"\n[√] 任务 [{desc}] 成功圆满完成！\n")

def find_best_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir) if f.startswith("CASE_") and not f.endswith(".json")]
    if not ckpts:
        return None
    # 提取最低 PPL 的检查点
    def parse_ppl(name):
        try:
            return float(name.split("_")[-1])
        except Exception:
            return 999.0
    ckpts.sort(key=parse_ppl)
    best = os.path.join(save_dir, ckpts[0])
    print(f"[Checkpoint Hunter] 在 {save_dir} 锁定最优黄金权重: {best}")
    return best

def clean_redundant_checkpoints(save_dir, best_ckpt_path):
    """物理销毁全部非最优检查点，仅留唯一黄金权重"""
    if not os.path.exists(save_dir):
        return
    for fname in os.listdir(save_dir):
        if fname.startswith("CASE_") and not fname.endswith(".json"):
            fpath = os.path.join(save_dir, fname)
            if os.path.abspath(fpath) != os.path.abspath(best_ckpt_path):
                try:
                    os.remove(fpath)
                    print(f"[Disk Maintenance] 自动物理清除历史冗余检查点: {fname}")
                except Exception as e:
                    print(f"[Disk Maintenance Warning] 清除失败 {fname}: {e}")

def run_pipeline():
    parser = argparse.ArgumentParser(description="CASE-EPCL V8.1 双轨对比实验自动化调度器")
    parser.add_argument("--skip_trial_a", action="store_true", help="是否跳过路线 A")
    parser.add_argument("--skip_trial_b", action="store_true", help="是否跳过路线 B")
    parser.add_argument("--only_eval", action="store_true", help="仅对已有权重执行评测")
    parser.add_argument("--resume_from_eval_a", action="store_true", help="从 Trial A 评测开始续跑，并无缝接续 Trial B 训练与评测")
    args = parser.parse_args()

    check_disk_space(10.0)

    # =========================================================================
    # 1. 路线 A (Trial A): 词表层情感偏置注入 + 全程联合微调
    # =========================================================================
    trial_a_dir = os.path.join(PROJECT_ROOT, "save", "epcl_v8_1_trialA")
    trial_a_log = os.path.join(PROJECT_ROOT, "save", "epcl_v8_1_trialA", "train.log")
    
    cmd_trial_a_train = [
        PYTHON_EXE, "main.py",
        "--dataset", "ED",
        "--woStrategy",
        "--batch_size", "8",
        "--pretrain",
        "--pretrain_epoch", "4",
        "--warmup", "24000",
        "--epcl_warmup", "6000",
        "--fine_weight", "0.3",
        "--coarse_weight", "1.0",
        "--seed", "13",
        "--gpu", "0",
        "--lr_schedule", "cosine",
        "--alpha_mim", "0.10",
        "--div_weight", "2.0",
        "--disable_freeze",
        "--emotion_head_type", "residual_mlp",
        "--mlp_hidden_dim", "300",
        "--mlp_dropout", "0.1",
        "--num_prototypes_per_class", "1",
        "--alpha_uni", "1.0",
        "--lambda_epcl", "0.08",
        "--use_pcam",
        "--pcam_heads", "2",
        "--pcam_dropout", "0.1",
        "--pcam_gate_bias", "-1.0",
        "--use_erp",
        "--erp_hidden_dim", "768",
        "--erp_dropout", "0.1",
        "--use_mcp",
        "--mcp_momentum", "0.99",
        "--use_arc_margin",
        "--arc_margin", "0.30",
        "--arc_mode", "cos",
        "--epcl_anchor", "emotion_enc",
        "--use_emo_bias",
        "--emo_bias_gate_init", "-2.0",
        "--use_unlikelihood",
        "--unlikelihood_weight", "0.1",
        "--save_path", "save/epcl_v8_1_trialA"
    ]

    if not args.skip_trial_a:
        if not args.only_eval and not args.resume_from_eval_a:
            run_command_with_logging(cmd_trial_a_train, trial_a_log, desc="V8.1-Trial A 训练 (词表偏置+联合微调)")
        
        # 评测与磁盘维护
        best_a = find_best_checkpoint(trial_a_dir)
        if best_a:
            clean_redundant_checkpoints(trial_a_dir, best_a)
            check_disk_space(10.0)
            
            cmd_trial_a_eval = [
                PYTHON_EXE, "src/scripts/eval_v8_pipeline.py",
                "--save_dir", "save/epcl_v8_1_trialA",
                "--epcl_anchor", "emotion_enc",
                "--use_arc_margin",
                "--arc_margin", "0.30",
                "--use_emo_bias",
                "--plot_path", "docs/v8/images/v8_1_trialA_manifold.png"
            ]
            eval_log_a = os.path.join(trial_a_dir, "eval_pipeline.log")
            run_command_with_logging(cmd_trial_a_eval, eval_log_a, desc="V8.1-Trial A 自动化指标度量与流形评测")

    # =========================================================================
    # 2. 路线 B (Trial B): 余弦软退火解耦 + 复合早停机制
    # =========================================================================
    trial_b_dir = os.path.join(PROJECT_ROOT, "save", "epcl_v8_1_trialB")
    trial_b_log = os.path.join(PROJECT_ROOT, "save", "epcl_v8_1_trialB", "train.log")

    cmd_trial_b_train = [
        PYTHON_EXE, "main.py",
        "--dataset", "ED",
        "--woStrategy",
        "--batch_size", "8",
        "--pretrain",
        "--pretrain_epoch", "4",
        "--warmup", "24000",
        "--epcl_warmup", "6000",
        "--fine_weight", "0.3",
        "--coarse_weight", "1.0",
        "--seed", "13",
        "--gpu", "0",
        "--lr_schedule", "cosine",
        "--alpha_mim", "0.10",
        "--div_weight", "2.0",
        "--disable_freeze",
        "--emotion_head_type", "residual_mlp",
        "--mlp_hidden_dim", "300",
        "--mlp_dropout", "0.1",
        "--num_prototypes_per_class", "1",
        "--alpha_uni", "1.0",
        "--lambda_epcl", "0.15",
        "--use_pcam",
        "--pcam_heads", "2",
        "--pcam_dropout", "0.1",
        "--pcam_gate_bias", "-1.0",
        "--use_erp",
        "--erp_hidden_dim", "768",
        "--erp_dropout", "0.1",
        "--use_mcp",
        "--mcp_momentum", "0.99",
        "--use_arc_margin",
        "--arc_margin", "0.30",
        "--arc_mode", "cos",
        "--epcl_anchor", "emotion_enc",
        "--use_cosine_anneal",
        "--anneal_start_step", "24000",
        "--anneal_steps", "16000",
        "--anneal_min_weight", "0.05",
        "--use_composite_score",
        "--composite_acc_weight", "20.0",
        "--use_unlikelihood",
        "--unlikelihood_weight", "0.1",
        "--save_path", "save/epcl_v8_1_trialB"
    ]

    if not args.skip_trial_b:
        if not args.only_eval:
            run_command_with_logging(cmd_trial_b_train, trial_b_log, desc="V8.1-Trial B 训练 (软退火+复合早停)")
        
        # 评测与磁盘维护
        best_b = find_best_checkpoint(trial_b_dir)
        if best_b:
            clean_redundant_checkpoints(trial_b_dir, best_b)
            check_disk_space(10.0)

            cmd_trial_b_eval = [
                PYTHON_EXE, "src/scripts/eval_v8_pipeline.py",
                "--save_dir", "save/epcl_v8_1_trialB",
                "--epcl_anchor", "emotion_enc",
                "--use_arc_margin",
                "--arc_margin", "0.30",
                "--plot_path", "docs/v8/images/v8_1_trialB_manifold.png"
            ]
            eval_log_b = os.path.join(trial_b_dir, "eval_pipeline.log")
            run_command_with_logging(cmd_trial_b_eval, eval_log_b, desc="V8.1-Trial B 自动化指标度量与流形评测")

    # =========================================================================
    # 3. 生成双轨横向学术对比报告
    # =========================================================================
    log_header("生成 V8.1 双轨实验横向对比学术报告")
    report_file = os.path.join(PROJECT_ROOT, "docs", "v8", "v8.1_dual_track_experiment_report.md")
    generate_comparison_report(trial_a_dir, trial_b_dir, report_file)
    print(f"[√] 双轨对比报告已生成: {report_file}")


def generate_comparison_report(dir_a, dir_b, out_path):
    import json
    
    def read_metrics(save_dir):
        json_path = os.path.join(save_dir, "v8_evaluation_summary.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    data_a = read_metrics(dir_a)
    data_b = read_metrics(dir_b)

    report_content = f"""# CASE-EPCL V8.1 双轨实验综合学术评估报告

## 一、双轨实验设计背景与学术假设

- **路线 A (Trial A)**: 词表层情感偏置注入 (KEMP Emotion-Aware Vocab Bias) + 全程联合微调 (disable_freeze)。
  - **核心假设**: 消除人为切断分类头梯度的硬性解耦，通过原型投影向解码器输出 logits 注入显式情感导向，在保护生成 PPL 与多样性的同时，彻底稳住多任务分类准确率 (EMO_acc >= 40.0%)。
- **路线 B (Trial B)**: 平滑余弦软退火解耦 (Cosine Soft Annealing) + 复合早停机制 (Composite Early Stopping)。
  - **核心假设**: 尊重多任务塑形与单任务生成阶段分离的直觉，在 24k~40k 步通过余弦平滑衰减辅助损失至 0.05（保持微弱梯度流以维持原型流形不崩溃），配合 $Score = PPL - 20 \\times EMO\\_acc$ 复合早停，追求极致生成质量 (PPL 32.30~32.50)。

---

## 二、全量学术指标横向对比总表

| 评估维度 | 指标项 | 基线 (V7-Best) | V8-Trial 3 | V8-Trial 4 | **V8.1-Trial A (偏置联合)** | **V8.1-Trial B (软退火解耦)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **语言生成质量** | **困惑度 (PPL) ↓** | 33.72 | 33.12 | 33.40 | **{data_a.get('Test_Set_Metrics', {}).get('Test_PPL', '待测')}** | **{data_b.get('Test_Set_Metrics', {}).get('Test_PPL', '待测')}** |
| | **Distinct-1 (%) ↑** | 8.85 | 8.52 | 8.65 | {data_a.get('Diversity_Metrics', {}).get('Greedy', {}).get('Dist-1', '待测')} | {data_b.get('Diversity_Metrics', {}).get('Greedy', {}).get('Dist-1', '待测')} |
| | **Distinct-2 (%) ↑** | 58.74 | 60.10 | 60.52 | {data_a.get('Diversity_Metrics', {}).get('Greedy', {}).get('Dist-2', '待测')} | {data_b.get('Diversity_Metrics', {}).get('Greedy', {}).get('Dist-2', '待测')} |
| **情感分类能力** | **EMO_acc (%) ↑** | 38.64 | 39.28 | 39.66 | **{data_a.get('Test_Set_Metrics', {}).get('Test_EMO_acc', '待测')}** | **{data_b.get('Test_Set_Metrics', {}).get('Test_EMO_acc', '待测')}** |
| | **EMO_loss ↓** | 2.5120 | 2.3967 | 2.8137 | {data_a.get('Test_Set_Metrics', {}).get('Test_EMO_loss', '待测')} | {data_b.get('Test_Set_Metrics', {}).get('Test_EMO_loss', '待测')} |
| **原型流形质量** | **对齐度 (Alignment) ↓** | 0.8124 | 0.7412 | 0.7250 | {data_a.get('Manifold_Metrics', {}).get('Alignment_Loss', '待测')} | {data_b.get('Manifold_Metrics', {}).get('Alignment_Loss', '待测')} |
| | **均匀度 (Uniformity) ↓** | -1.5420 | -1.8210 | -1.8540 | {data_a.get('Manifold_Metrics', {}).get('Uniformity_Loss', '待测')} | {data_b.get('Manifold_Metrics', {}).get('Uniformity_Loss', '待测')} |
| | **轮廓系数 (Silhouette) ↑** | 0.0842 | 0.1120 | 0.1245 | {data_a.get('Manifold_Metrics', {}).get('Silhouette_Score', '待测')} | {data_b.get('Manifold_Metrics', {}).get('Silhouette_Score', '待测')} |
| | **DB 指数 (Davies-Bouldin) ↓**| 2.8541 | 2.5120 | 2.4501 | {data_a.get('Manifold_Metrics', {}).get('Davies_Bouldin_Index', '待测')} | {data_b.get('Manifold_Metrics', {}).get('Davies_Bouldin_Index', '待测')} |

---

## 三、流形可视化与结构分析

### 1. 路线 A 原型与特征流形
![路线 A 流形图](images/v8_1_trialA_manifold.png)

### 2. 路线 B 原型与特征流形
![路线 B 流形图](images/v8_1_trialB_manifold.png)

---

## 四、核心学术发现与理论启示

1. **多任务协同 vs 阶段解耦**：
   - 深入分析词表层偏置是否在保持生成多样性的同时，提供了足够强的共情偏置。
   - 验证余弦平滑软退火是否成功克服了早期版本硬冻结带来的泛化断崖与流形畸变。
2. **论文最终推荐选型**：
   - 根据生成质量与共情评估，选出 V8.1 的主力交付架构，作为最终 ACL/EMNLP 投件标准模型。
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    run_pipeline()
