#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 核心生产级训练与推断启动脚本 (main.sh)
# 针对 24GB 显存 (RTX 4090 / 3090) 优化，支持单卡大 Batch 训练与自适应时序调度
# ==============================================================================
set -e

# Python 解释器路径 (根据实际虚拟环境配置，例如 conda 环境或系统 python)
pythonpath='python'

# 1. 核心硬件运行模式一键切换
# 支持命令行指定: ./main.sh 24G (默认生产全量) 或 ./main.sh 4G (本地轻量调试)
ENV_MODE=${1:-"24G"}

DATASET='ED'
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}
SEED=13
PRETRAIN_EPOCH=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$ENV_MODE" = "4G" ]; then
    echo "======================================================================"
    echo "[*] 加载 4GB 显存轻量模式 (RTX 3050Ti 本地兼容调试)"
    echo "    - 特性: FP16 混合精度 + 梯度累加 4 次 (等效 Batch 32) + 显存占用 < 3.8GB"
    echo "======================================================================"
    BATCH_SIZE=8
    ACCUM_STEPS=4
    PRECISION="fp16"
    LR=0.0001
    WARMUP=12000
    OUTPUT_DIR="save/v9_4g_debug/"
else
    echo "======================================================================"
    echo "[*] 加载 24GB 显存生产级全量模式 (RTX 4090 / 3090 旗舰训练)"
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 32 x 累加 2 (等效 Batch 64) + 显存安全可控"
    echo "======================================================================"
    # 【方案 2: 当前激活】等效 Batch 64 (单步物理 32 x 累加 2，显存峰值约 12GB，兼顾大批次稳定性与防 OOM)
    BATCH_SIZE=32
    ACCUM_STEPS=2
    PRECISION="fp32"  # 4090 也可换为 bf16
    LR=0.0003
    WARMUP=2000
    OUTPUT_DIR="save/v9_24g_baseline/"

    # 【方案 1: 备用注释】100% 还原原作者 ACL 2023 官方默认基准 (单步物理 16，零累加，显存峰值约 7GB)
    # BATCH_SIZE=16
    # ACCUM_STEPS=1
    # PRECISION="fp32"
    # LR=0.0001
    # WARMUP=4000
    # OUTPUT_DIR="save/v9_24g_baseline_orig16/"
fi

# 2. 执行多任务训练
mkdir -p logs
LOG_FILE="logs/train.log"

${pythonpath} main.py \
  --dataset ${DATASET} \
  --gpu ${GPU_ID} \
  --seed ${SEED} \
  --batch_size ${BATCH_SIZE} \
  --accum_steps ${ACCUM_STEPS} \
  --precision ${PRECISION} \
  --lr ${LR} \
  --warmup ${WARMUP} \
  --pretrain \
  --pretrain_epoch ${PRETRAIN_EPOCH} \
  --woStrategy \
  --fine_weight 0.2 \
  --coarse_weight 1.0 \
  --use_mcp \
  --mcp_momentum 0.96 \
  --lambda_epcl 0.07 \
  --arc_margin 0.30 \
  --arc_mode cos \
  --epcl_anchor fine_emotion \
  --cls_anchor default \
  --use_sparse_emo_bias \
  --emo_vocab_topk_ratio 0.15 \
  --emo_bias_gate_init -2.0 \
  --use_pcgrad \
  --gate_warmup_steps 3000 \
  --mask_update_interval 1000 \
  --use_bias_annealing \
  --bias_anneal_start 5000 \
  --bias_anneal_steps 3000 \
  --bias_min_scale 0.10 \
  --use_emo_loss_ramp \
  --emo_loss_ramp_start 3500 \
  --emo_loss_ramp_steps 3000 \
  --emo_loss_ramp_max 1.30 \
  --emo_loss_ramp_shape bell \
  --use_composite_score \
  --composite_mode emo_loss \
  --composite_emo_loss_weight 5.0 \
  --min_save_step 4500 \
  --patience 10 \
  --save_path ${OUTPUT_DIR} \
  --model_file_path ${OUTPUT_DIR} 2>&1 | tee ${LOG_FILE}

echo "======================================================================"
echo "[*] 训练完成！自动启动全量学术指标自动化审计..."
echo "======================================================================"

# 自动调用统一评测流水线进行测试集度量 (PPL, Dist-2, Unique, Alignment, DBI)
${pythonpath} src/scripts/eval_pipeline.py \
  --model_path ${OUTPUT_DIR}CASE_best.pth \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU_ID}
