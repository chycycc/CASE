# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.2 全自动化实战执行流水线 (V8.2 Automated Pipeline Runner)

功能涵盖:
1. 环境与磁盘安全审计 (RTX 3050 4GB 显存 / E 盘 > 10GB 空间审计)
2. 启动 V8.2 破局实战训练:
   - 原型-词嵌入拓扑亲和度掩码 (自适应稀疏化 Top 15%, 85% 功能词 Zero-Mask)
   - 多任务正交梯度投影 (PCGrad 消除对抗性负点积)
   - 动态门控时序平滑调度 (前 20k 步冷启动置 -5.0)
   - 全程端到端联合微调 (disable_freeze 稳守 40% 分类基准)
3. 单黄金检查点自动追踪与多余权重物理销毁 (维持磁盘纯净)
4. 自动化双轨解码评测流水线 (eval_v8_pipeline.py: 贪婪 + 核采样 + 流形几何聚类)
5. 自动提取核心指标并输出学术报告
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
            if any(k in line for k in ["[Adaptive", "[*] Step", "[-] Step", "Epoch", "Val PPL", "loss_train", "results", "[+]"]):
                safe_print(f"[{desc}] {line.strip()}")
                last_print_time = curr_time
            elif curr_time - last_print_time > 60.0:
                safe_print(f"[{desc} 运行心跳] 正在平稳训练中... (最新行: {line.strip()[:60]}...)")
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
    def parse_ppl(name):
        try:
            return float(name.split("_")[-1])
        except Exception:
            return 999.0
    ckpts.sort(key=parse_ppl)
    best = os.path.join(save_dir, ckpts[0])
    safe_print(f"[Checkpoint Hunter] 在 {save_dir} 锁定最优黄金权重: {best}")
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
                    safe_print(f"[Disk Maintenance] 自动物理清除冗余中间权重: {fname}")
                except Exception as e:
                    safe_print(f"[Disk Maintenance Warning] 清除失败 {fname}: {e}")

def main():
    parser = argparse.ArgumentParser(description="CASE-EPCL V8.2 全自动化流水线")
    parser.add_argument("--only_eval", action="store_true", help="仅执行评测流水线")
    parser.add_argument("--skip_eval", action="store_true", help="仅执行训练流水线")
    args = parser.parse_args()

    v8_2_dir = os.path.join(PROJECT_ROOT, "save", "epcl_v8_2")
    os.makedirs(v8_2_dir, exist_ok=True)
    train_log = os.path.join(v8_2_dir, "train.log")
    eval_log = os.path.join(v8_2_dir, "eval_pipeline.log")

    log_header("CASE-EPCL V8.2 破局全自动执行流水线启动")
    check_disk_space(10.0)

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
        "--unlikelihood_weight", "0.1",
        "--save_path", "save/epcl_v8_2"
    ]

    # 执行训练
    if not args.only_eval:
        run_command_with_logging(cmd_train, train_log, desc="V8.2 破局架构正式实战训练")

    # 磁盘维护与黄金权重查找
    best_ckpt = find_best_checkpoint(v8_2_dir)
    if best_ckpt:
        clean_redundant_checkpoints(v8_2_dir, best_ckpt)
        check_disk_space(10.0)

    # 2. 评测流水线执行
    if not args.skip_eval and best_ckpt:
        os.makedirs(os.path.join(PROJECT_ROOT, "docs", "v8", "images"), exist_ok=True)
        cmd_eval = [
            PYTHON_EXE, "src/scripts/eval_v8_pipeline.py",
            "--save_dir", "save/epcl_v8_2",
            "--epcl_anchor", "emotion_enc",
            "--use_arc_margin",
            "--arc_margin", "0.30",
            "--use_emo_bias",
            "--use_sparse_emo_bias",
            "--emo_vocab_topk_ratio", "0.15",
            "--plot_path", "docs/v8/images/v8_2_manifold.png"
        ]
        run_command_with_logging(cmd_eval, eval_log, desc="V8.2 自动化学术指标度量与流形评测")

    log_header("V8.2 全自动化流水线顺利执行完毕！")

if __name__ == "__main__":
    main()
