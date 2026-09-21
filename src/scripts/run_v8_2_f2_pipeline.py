# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.2 F2 快速验证流水线 (α=7.0 帕累托检查点策略调优)

与 V8.2 F 唯一区别:
- --composite_emo_loss_weight 从 5.0 提升至 7.0
- 目标: 让 Step 33,999 (EMO_loss=2.1493) 胜出而非 Step 39,999

训练过程完全相同，仅检查点选择策略不同。
"""

import os
import sys
import time
import shutil
import subprocess
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

def check_disk_space(min_gb=8.0):
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
            cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env
        )
        last_print_time = time.time()
        for line in process.stdout:
            f_log.write(line)
            f_log.flush()
            curr_time = time.time()
            if any(k in line for k in ["[Adaptive", "[*] Step", "[-] Step", "Epoch", "Val PPL", "loss_train", "results", "[+]", "Early Stopping", "Score=", "[V8.2"]):
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
    mature_ckpts = [c for c in ckpts if c["step"] >= 30000]
    if mature_ckpts:
        mature_ckpts.sort(key=lambda x: x["ppl"])
        return mature_ckpts[0]
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
    safe_print(f"[Disk Cleanup] 当前唯一成熟期黄金检查点: {os.path.basename(best_ckpt_info['path'])} (Val PPL: {best_ckpt_info['ppl']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="CASE-EPCL V8.2 F2 快速验证 (α=7.0)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    log_header("CASE-EPCL V8.2 F2 (α=7.0) 帕累托检查点策略验证启动")
    check_disk_space(8.0)

    save_dir = os.path.join(PROJECT_ROOT, "save", "epcl_v8_2_f2")
    os.makedirs(save_dir, exist_ok=True)
    train_log = os.path.join(save_dir, "train.log")
    eval_log = os.path.join(save_dir, "eval_pipeline.log")

    # 训练命令 (与 V8.2 F 完全一致，仅 α=7.0)
    cmd_train = [
        PYTHON_EXE, "main.py",
        "--dataset", "ED", "--woStrategy", "--batch_size", "8",
        "--pretrain", "--pretrain_epoch", "4",
        "--warmup", "24000", "--epcl_warmup", "6000",
        "--fine_weight", "0.3", "--coarse_weight", "1.0",
        "--seed", "13", "--gpu", "0",
        "--lr_schedule", "cosine",
        "--anneal_start_step", "24000", "--anneal_steps", "20000",
        "--patience", "10",
        "--alpha_mim", "0.10", "--div_weight", "2.0",
        "--disable_freeze",
        "--emotion_head_type", "residual_mlp", "--mlp_hidden_dim", "300", "--mlp_dropout", "0.1",
        "--num_prototypes_per_class", "1", "--alpha_uni", "1.0", "--lambda_epcl", "0.08",
        "--use_pcam", "--pcam_heads", "2", "--pcam_dropout", "0.1", "--pcam_gate_bias", "-1.0",
        "--use_erp", "--erp_hidden_dim", "768", "--erp_dropout", "0.1",
        "--use_mcp", "--mcp_momentum", "0.99",
        "--use_arc_margin", "--arc_margin", "0.30", "--arc_mode", "cos",
        "--epcl_anchor", "emotion_enc", "--cls_anchor", "emotion_enc",
        "--use_emo_bias", "--emo_bias_gate_init", "-5.0",
        "--use_sparse_emo_bias", "--emo_vocab_topk_ratio", "0.15",
        "--use_pcgrad",
        "--gate_warmup_steps", "20000", "--mask_update_interval", "5000",
        "--use_unlikelihood", "--unlikelihood_weight", "0.05",
        "--use_bias_annealing", "--bias_anneal_start", "24000",
        "--bias_anneal_steps", "10000", "--bias_min_scale", "0.10",
        "--use_emo_loss_ramp", "--emo_loss_ramp_start", "24000",
        "--emo_loss_ramp_steps", "16000", "--emo_loss_ramp_max", "1.3",
        "--emo_loss_ramp_shape", "bell",
        "--min_save_step", "30000",
        "--use_composite_score", "--composite_mode", "emo_loss",
        # ===== 唯一变更: α 从 5.0 提升至 7.0 =====
        "--composite_emo_loss_weight", "7.0",
        "--save_path", "save/epcl_v8_2_f2"
    ]

    if not args.only_eval and not args.skip_train:
        run_command_with_logging(cmd_train, train_log, desc="V8.2 F2 (α=7.0 帕累托检查点策略验证)")

    best_ckpt = find_best_checkpoint(save_dir)
    if best_ckpt:
        clean_redundant_checkpoints(save_dir, best_ckpt)
        check_disk_space(8.0)

    if not args.skip_eval and best_ckpt:
        os.makedirs(os.path.join(PROJECT_ROOT, "docs", "v8", "images"), exist_ok=True)
        cmd_eval = [
            PYTHON_EXE, "src/scripts/eval_v8_pipeline.py",
            "--save_dir", "save/epcl_v8_2_f2",
            "--epcl_anchor", "emotion_enc", "--cls_anchor", "emotion_enc",
            "--use_arc_margin", "--arc_margin", "0.30",
            "--use_emo_bias", "--use_sparse_emo_bias", "--emo_vocab_topk_ratio", "0.15",
            "--bias_min_scale", "0.10",
            "--plot_path", "docs/v8/images/v8_2_f2_manifold.png"
        ]
        run_command_with_logging(cmd_eval, eval_log, desc="V8.2 F2 全量测试集评测")

    log_header("CASE-EPCL V8.2 F2 全自动化流水线执行完毕！")

if __name__ == "__main__":
    main()
