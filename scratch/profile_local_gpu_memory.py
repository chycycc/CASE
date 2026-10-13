# -*- coding: utf-8 -*-
"""
本地 4GB GPU (RTX 3050 Ti Laptop) 运行环境与显存/算力实测探针
用于评估 CASE-EPCL 实验在本地硬件上的运行可行性与吞吐率
"""
import os
import sys
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq

def profile_local_env():
    print("=" * 65)
    print("1. 本地硬件与 PyTorch 环境基础信息")
    print("=" * 65)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用性: {cuda_available}")
    if not cuda_available:
        print("未检测到 CUDA，退出。")
        return

    device_name = torch.cuda.get_device_name(0)
    total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"GPU 设备: {device_name}")
    print(f"总物理显存: {total_mem_mb:.2f} MB ({total_mem_mb / 1024:.2f} GB)")
    print(f"PyTorch 版本: {torch.__version__}, CUDA 版本: {torch.version.cuda}")

    # 2. 模型与配置准备
    config.seed = 13
    set_seed()
    config.device = torch.device("cuda:0")
    config.cuda = True
    config.pretrain = False
    config.use_pcam = True
    config.pcam_tau_attn = 1.5
    config.pcam_heads = 2
    config.pcam_gate_bias = -1.0
    config.use_mcp = True
    config.mcp_momentum = 0.96
    config.lambda_epcl = 0.03
    config.alpha_uni = 0.8
    config.arc_margin = 0.30
    config.arc_mode = "cos"
    config.use_arc_margin = True

    print("\n" + "=" * 65)
    print("2. 静态模型载入显存测试 (Model Parameter Footprint)")
    print("=" * 65)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=32)
    model = CASE(vocab, emotion_num=emo_num, strategy_num=strategy_num, is_eval=False)
    model = model.to(config.device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    static_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    static_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"模型总参数量: {param_count:,} ({param_count * 4 / (1024**2):.2f} MB FP32)")
    print(f"可训练参数量: {trainable_count:,}")
    print(f"模型加载后已分配显存: {static_alloc_mb:.2f} MB")
    print(f"模型加载后保留显存: {static_reserved_mb:.2f} MB")

    # 3. 动态训练前向+反向+优化步测试 (动态梯度与激活值显存)
    print("\n" + "=" * 65)
    print("3. 不同 Batch Size 下训练单步实测 (Forward + Backward + Step)")
    print("=" * 65)

    batch_sizes = [32, 16, 8]
    profile_results = {}

    for bs in batch_sizes:
        print(f"\n[*] 测试 batch_size = {bs} ...")
        # 重新准备 dataloader
        config.batch_size = bs
        train_loader, _, _, _, _, _ = prepare_data_seq(batch_size=bs)
        batch = next(iter(train_loader))

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()

        # 预热 1 步
        try:
            _ = model.train_one_batch(batch, iter=0, train=True)
            # 正式测量 3 步平均
            latencies = []
            for step in range(3):
                t0 = time.time()
                _ = model.train_one_batch(batch, iter=step+1, train=True)
                torch.cuda.synchronize()
                latencies.append(time.time() - t0)

            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            avg_step_sec = sum(latencies) / len(latencies)
            eta_10k_hours = (avg_step_sec * 10000) / 3600

            print(f"    - 显存峰值 (Allocated): {peak_alloc_mb:.2f} MB ({peak_alloc_mb / 1024:.2f} GB)")
            print(f"    - 显存保留 (Reserved):  {peak_reserved_mb:.2f} MB ({peak_reserved_mb / 1024:.2f} GB)")
            print(f"    - 显存余量:             {total_mem_mb - peak_reserved_mb:.2f} MB")
            print(f"    - 单步耗时:             {avg_step_sec * 1000:.1f} ms")
            print(f"    - 10k iter 预估总耗时:   {eta_10k_hours:.2f} 小时 ({eta_10k_hours * 60:.1f} 分钟)")
            profile_results[bs] = {
                "peak_alloc_mb": peak_alloc_mb,
                "peak_reserved_mb": peak_reserved_mb,
                "avg_step_sec": avg_step_sec,
                "eta_10k_hours": eta_10k_hours,
                "status": "PASS"
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [!] OOM (显存溢出): {e}")
                profile_results[bs] = {"status": "OOM"}
                torch.cuda.empty_cache()
            else:
                raise e

    # 4. 推理评测测试
    print("\n" + "=" * 65)
    print("4. 推理评测显存实测 (Batch=1 单样本解码)")
    print("=" * 65)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()

    test_loader = test_set
    sample = next(iter(test_loader))
    t0 = time.time()
    with torch.no_grad():
        _ = model.train_one_batch(sample, iter=0, train=False)
    torch.cuda.synchronize()
    eval_step_sec = time.time() - t0
    peak_eval_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_eval_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"评估单步显存峰值: {peak_eval_alloc_mb:.2f} MB")
    print(f"5,255 样本测试集前向耗时预估: {(eval_step_sec * 5255):.1f} 秒 ({(eval_step_sec * 5255)/60:.1f} 分钟)")

    print("\n" + "=" * 65)
    print("5. 综合可行性判决")
    print("=" * 65)
    bs32_res = profile_results.get(32, {})
    if bs32_res.get("status") == "PASS":
        print(f"[结论] 本地 4GB GPU 完全可以运行 batch_size=32 的完整训练！")
        print(f"显存峰值仅占 {bs32_res['peak_reserved_mb']:.1f} MB / {total_mem_mb:.1f} MB ({bs32_res['peak_reserved_mb']/total_mem_mb*100:.1f}%)。")
    else:
        print("[结论] 存在显存压力，需梯度累积。")

if __name__ == "__main__":
    profile_local_env()
