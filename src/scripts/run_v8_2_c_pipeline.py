# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.2 C 全自动化实战执行流水线 (V8.2 C Automated Pipeline Runner)

核心功能:
1. 磁盘安全与显存环境审计 (E 盘 > 10GB 空间审计 / RTX 3050 4GB 显存保护)
2. 启动 V8.2 C 全指标达标训练:
   - [对称双向正交投影] 标准对称 PCGrad 消除情感梯度剥夺，锁定 EMO_loss <= 2.1500
   - [偏置门控时序余弦退火] 35,000 步后余弦衰减至 0.2 下限，消除后程概率扰动，攻克 PPL <= 32.50
   - [序列级无似然降权] --unlikelihood_weight 0.05 减负语言建模 PPL 负担
   - [早停与收敛窗口扩大] --patience 15 保证 24k~44k 退火后在 1e-5 超低学习率下平稳收敛
   - 原型-词嵌入拓扑亲和度掩码 (自适应稀疏化 Top 15%, 85% 功能词 Zero-Mask)
   - 动态门控时序预热调度 (前 20k 步冷启动置 -5.0)
   - 全程端到端联合微调 (disable_freeze 稳守 40% 分类基准)
3. 严格执行单黄金检查点追踪与多余冗余权重物理销毁 (维持 E 盘空间纯净)
4. 自动化双轨解码与流形几何学术评测 (eval_v8_pipeline.py: Greedy + Beam + Sampling 0.7/0.8 + 质心对齐)
5. 自动提取核心指标并留档汇报
"""

import os
import sys
import time
import shutil
import subprocess
import argparse

# 注册工程根路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 控制台编码防乱码配置
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
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024 ** 3)
        safe_print(f"[Disk Audit] E 盘可用空间: {free_gb:.2f} GB (最低红线: {min_gb:.1f} GB)")
        if free_gb < min_gb:
            safe_print(f"[!] 警告: 磁盘可用空间低于 {min_gb} GB，请立即清理！")
        return free_gb
    except Exception as e:
        safe_print(f"[Disk Audit Warning] 无法查询磁盘容量: {e}")
        return 999.0

def run_command_with_logging(cmd, log_file_path, desc="任务"):
    log_header(f"启动流水线子任务: {desc}")
    safe_print(f"执行命令: {' '.join(cmd)}")
    safe_print(f"日志重定向: {log_file_path}\n")

    os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

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
            
            curr_time = time.time()
            if any(k in line for k in ["[Adaptive", "[*] Step", "[-] Step", "Epoch", "Val PPL", "loss_train", "results", "[+]", "Early Stopping"]):
                safe_print(f"[{desc}] {line.strip()}")
                last_print_time = curr_time
            elif curr_time - last_print_time > 60:
                safe_print(f"[{desc} 心跳] 任务正常运行中... 当前最新输出: {line.strip()[:100]}")
                last_print_time = curr_time

        process.wait()
        return_code = process.returncode

    if return_code != 0:
        safe_print(f"[!] 子任务执行失败 (Exit Code: {return_code})！详见日志: {log_file_path}")
        sys.exit(return_code)
    else:
        safe_print(f"[+] 子任务执行成功: {desc}\n")

def find_best_checkpoint(save_dir):
    import glob
    if not os.path.exists(save_dir):
        return None
    files = [f for f in glob.glob(os.path.join(save_dir, "CASE_*")) if not f.endswith(".json") and not f.endswith(".txt")]
    if not files:
        return None
    
    ckpts = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        try:
            ppl_val = float(parts[2])
            step_val = int(parts[1])
            ckpts.append({"path": f, "step": step_val, "ppl": ppl_val})
        except Exception:
            continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x["ppl"])
    return ckpts[0]

def clean_redundant_checkpoints(save_dir, best_ckpt_info):
    import glob
    files = [f for f in glob.glob(os.path.join(save_dir, "CASE_*")) if not f.endswith(".json") and not f.endswith(".txt")]
    removed = 0
    reclaimed_mb = 0.0
    for f in files:
        if os.path.abspath(f) != os.path.abspath(best_ckpt_info["path"]):
            try:
                sz = os.path.getsize(f) / (1024 * 1024)
                os.remove(f)
                removed += 1
                reclaimed_mb += sz
            except Exception as e:
                safe_print(f"  [!] 删除冗余权重失败: {f}, {e}")
    safe_print(f"[Disk Cleanup] 成功物理销毁 {removed} 个冗余权重，释放磁盘空间: {reclaimed_mb:.2f} MB")
    safe_print(f"[Disk Cleanup] 当前唯一黄金检查点: {os.path.basename(best_ckpt_info['path'])} (Val PPL: {best_ckpt_info['ppl']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="CASE-EPCL V8.2 C 全自动化流水线")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练阶段，直接执行评估")
    parser.add_argument("--only_eval", action="store_true", help="仅执行评测与报告生成")
    parser.add_argument("--skip_eval", action="store_true", help="仅训练，跳过测试集评测")
    args = parser.parse_args()

    log_header("CASE-EPCL V8.2 C 自动化执行流水线启动")
    check_disk_space(10.0)

    v8_2_c_dir = os.path.join(PROJECT_ROOT, "save", "epcl_v8_2_c")
    os.makedirs(v8_2_c_dir, exist_ok=True)
    train_log = os.path.join(v8_2_c_dir, "train.log")
    eval_log = os.path.join(v8_2_c_dir, "eval_pipeline.log")

    # 1. 训练命令配置
    cmd_train = [
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
        "--anneal_start_step", "24000",
        "--anneal_steps", "20000",
        "--patience", "15",
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
        "--emo_bias_gate_init", "-5.0",
        "--use_sparse_emo_bias",
        "--emo_vocab_topk_ratio", "0.15",
        "--use_pcgrad",
        "--gate_warmup_steps", "20000",
        "--mask_update_interval", "5000",
        "--use_unlikelihood",
        "--unlikelihood_weight", "0.05",
        "--use_bias_annealing",
        "--bias_anneal_start", "35000",
        "--bias_anneal_steps", "15000",
        "--bias_min_scale", "0.2",
        "--save_path", "save/epcl_v8_2_c"
    ]

    # 执行训练
    if not args.only_eval and not args.skip_train:
        run_command_with_logging(cmd_train, train_log, desc="V8.2 C 全指标达标实战训练 (对称PCGrad + 偏置余弦退火 + UL0.05)")

    # 磁盘维护与黄金权重查找
    best_ckpt = find_best_checkpoint(v8_2_c_dir)
    if best_ckpt:
        clean_redundant_checkpoints(v8_2_c_dir, best_ckpt)
        check_disk_space(10.0)

    # 2. 评测流水线执行
    if not args.skip_eval and best_ckpt:
        os.makedirs(os.path.join(PROJECT_ROOT, "docs", "v8", "images"), exist_ok=True)
        cmd_eval = [
            PYTHON_EXE, "src/scripts/eval_v8_pipeline.py",
            "--save_dir", "save/epcl_v8_2_c",
            "--epcl_anchor", "emotion_enc",
            "--use_arc_margin",
            "--arc_margin", "0.30",
            "--use_emo_bias",
            "--use_sparse_emo_bias",
            "--emo_vocab_topk_ratio", "0.15",
            "--plot_path", "docs/v8/images/v8_2_c_manifold.png"
        ]
        run_command_with_logging(cmd_eval, eval_log, desc="V8.2 C 全量测试集学术指标度量与多温度评测")

    log_header("CASE-EPCL V8.2 C 全自动化流水线顺利执行完毕！")

if __name__ == "__main__":
    main()
