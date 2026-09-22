#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 核心生产级训练与推断启动脚本 (main.sh)
# 针对 24GB 显存 (RTX 4090 / 3090) 优化，支持单卡大 Batch 训练与自适应时序调度
# ==============================================================================
set -e

# Python 解释器路径 (根据实际虚拟环境配置，例如 conda 环境或系统 python)
pythonpath='python'

# 1. 核心硬件运行模式与实验代号支持
# 支持命令行指定: ./main.sh 24G v9_trial2 (默认生产全量) 或 ./main.sh 4G v9_debug_4g
ENV_MODE=${1:-"24G"}
EXP_NAME=${2:-"v9_trial2"}

DATASET='ED'
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}
SEED=13
PRETRAIN_EPOCH=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$ENV_MODE" = "4G" ]; then
    echo "======================================================================"
    echo "[*] 加载 4GB 显存轻量模式 (RTX 3050Ti 本地兼容调试)"
    echo "    - 实验代号: ${EXP_NAME}"
    echo "    - 特性: FP16 混合精度 + 梯度累加 4 次 (等效 Batch 32) + 显存占用 < 3.8GB"
    echo "======================================================================"
    BATCH_SIZE=8
    ACCUM_STEPS=4
    PRECISION="fp16"
    LR=0.0001
    WARMUP=12000
    CHECK_ITER=2000
    MAX_STEP=40000
    MIN_SAVE_STEP=16000
    PATIENCE=10
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"
else
    echo "======================================================================"
    echo "[*] 加载 24GB 显存生产级全量模式 (RTX 4090 / 3090 旗舰训练)"
    echo "    - 实验代号: ${EXP_NAME}"
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 32 x 累加 2 (等效 Batch 64) + 显存安全可控"
    echo "======================================================================"
    # 【方案 2: 当前激活】等效 Batch 64 (单步物理 32 x 累加 2，显存峰值约 12GB，兼顾大批次稳定性与防 OOM)
    # [V9 大批次尺度折算]: ED 数据集在 Batch 64 下每轮仅约 305 步，10000 步相当于约 32 个 Epoch
    BATCH_SIZE=32
    ACCUM_STEPS=2
    PRECISION="fp32"  # 4090 也可换为 bf16
    LR=0.0003
    WARMUP=2000
    CHECK_ITER=500
    MAX_STEP=10000
    MIN_SAVE_STEP=3000
    PATIENCE=6
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"

    # 【方案 1: 备用注释】100% 还原原作者 ACL 2023 官方默认基准 (单步物理 16，零累加，显存峰值约 7GB)
    # BATCH_SIZE=16
    # ACCUM_STEPS=1
    # PRECISION="fp32"
    # LR=0.0001
    # WARMUP=4000
    # CHECK_ITER=1000
    # MAX_STEP=20000
    # MIN_SAVE_STEP=6000
    # PATIENCE=8
    # OUTPUT_DIR="save/${EXP_NAME}_orig16/"
    # LOG_FILE="logs/${EXP_NAME}_orig16.log"
fi

# 2. 建立独立日志与实验产出目录
mkdir -p logs
mkdir -p results/v9

echo "[*] 启动配置全览:"
echo "    - 实验代号: ${EXP_NAME}"
echo "    - 权重保存: ${OUTPUT_DIR}"
echo "    - 训练日志: ${LOG_FILE}"
echo "    - 评估间隔: 每 ${CHECK_ITER} 步 | 最大步数: ${MAX_STEP} 步 | 保护期: ${MIN_SAVE_STEP} 步"
echo "======================================================================"

${pythonpath} main.py \
  --dataset ${DATASET} \
  --gpu ${GPU_ID} \
  --seed ${SEED} \
  --exp_name ${EXP_NAME} \
  --batch_size ${BATCH_SIZE} \
  --accum_steps ${ACCUM_STEPS} \
  --precision ${PRECISION} \
  --lr ${LR} \
  --warmup ${WARMUP} \
  --check_iter ${CHECK_ITER} \
  --max_step ${MAX_STEP} \
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
  --min_save_step ${MIN_SAVE_STEP} \
  --patience ${PATIENCE} \
  --save_path ${OUTPUT_DIR} \
  --model_file_path ${OUTPUT_DIR} 2>&1 | tee ${LOG_FILE}

echo "======================================================================"
echo "[*] 训练完成！自动启动全量学术指标自动化审计..."
echo "======================================================================"

# 自动调用统一评测流水线进行测试集度量 (PPL, Dist-2, Unique, Alignment, DBI)
${pythonpath} src/scripts/eval_pipeline.py \
  --model_path ${OUTPUT_DIR} \
  --results_path "results/v9/${EXP_NAME}_eval.txt" \
  --plot_path "results/v9/${EXP_NAME}_manifold.png" \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU_ID}
