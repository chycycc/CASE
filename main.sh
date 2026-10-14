#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 核心生产级训练与推断启动脚本 (main.sh)
# 针对 24GB 显存 (RTX 4090 / 3090) 优化，支持单卡大 Batch 训练与自适应时序调度
# ==============================================================================
set -e
set -o pipefail

# 智能匹配 Python 解释器路径 (支持环境变量注入，优先适配本地 conda cem_env，回退系统 python)
if [ -n "$PYTHON_BIN" ]; then
    pythonpath="$PYTHON_BIN"
elif [ -f "/e/Dev/condaData/envs_dirs/cem_env/python.exe" ]; then
    pythonpath="/e/Dev/condaData/envs_dirs/cem_env/python.exe"
elif [ -f "E:/Dev/condaData/envs_dirs/cem_env/python.exe" ]; then
    pythonpath="E:/Dev/condaData/envs_dirs/cem_env/python.exe"
else
    pythonpath="python"
fi

# 1. 核心硬件运行模式与实验代号支持
# 支持命令行指定: ./main.sh 4G v9_trial10 (默认本地兼容) 或 ./main.sh 24G v9_trial10 (云端生产全量)
ENV_MODE=${1:-"4G"}
EXP_NAME=${2:-"v9_trial10"}

DATASET='ED'
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}
SEED=13
PRETRAIN_EPOCH=4

# PyTorch 显存分配器版本自适应 (PyTorch 2.1+ 支持 expandable_segments，老版本如 1.12 使用 max_split_size_mb)
if ${pythonpath} -c "import torch; v=torch.__version__.split('+')[0].split('.'); exit(0 if (int(v[0]), int(v[1])) >= (2, 1) else 1)" 2>/dev/null; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
fi

if [ "$ENV_MODE" = "4G" ]; then
    echo "======================================================================"
    echo "[*] 加载 4GB 显存轻量模式 (RTX 3050Ti 本地兼容训练)"
    echo "    - 实验代号: ${EXP_NAME}"
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 8 梯度累加 4 次 (等效 Batch 32) + 显存峰值 < 1.4GB"
    echo "======================================================================"
    BATCH_SIZE=8
    ACCUM_STEPS=4
    PRECISION="fp32"
    LR=0.000576
    WARMUP=2000
    CHECK_ITER=500
    MAX_STEP=10000
    MIN_SAVE_STEP=3000
    PATIENCE=6
    EXTRA_FLAGS=""
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"
else
    echo "======================================================================"
    echo "[*] 加载 24GB 显存生产级全量模式 (RTX 4090 / 3090 旗舰训练)"
    echo "    - 实验代号: ${EXP_NAME}"
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 32 零累加 (更新步数 10,000 次) + 显存安全可控"
    echo "======================================================================"
    # 【Trial 8 激活】单步物理 32 x 累加 1 (零累加)，参数更新次数恢复为 10,000 次，提升步频分辨率
    BATCH_SIZE=32
    ACCUM_STEPS=1
    PRECISION="fp32"  # 4090 也可换为 bf16
    LR=0.000576
    WARMUP=2000
    CHECK_ITER=500
    MAX_STEP=10000
    MIN_SAVE_STEP=3000
    PATIENCE=6
    EXTRA_FLAGS=""
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
echo "    - 硬件模式: ${ENV_MODE} (Batch: ${BATCH_SIZE}, Accum: ${ACCUM_STEPS}, Precision: ${PRECISION})"
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
  --lr_schedule cosine \
  --lr_decay_start_step 3000 \
  --lr_decay_steps 6000 \
  --check_iter ${CHECK_ITER} \
  --max_step ${MAX_STEP} \
  --pretrain \
  --pretrain_epoch ${PRETRAIN_EPOCH} \
  --woStrategy \
  --fine_weight 0.2 \
  --coarse_weight 1.0 \
  --use_mcp \
  --mcp_momentum 0.96 \
  --lambda_epcl 0.03 \
  --alpha_uni 0.8 \
  --use_arc_margin \
  --arc_margin 0.30 \
  --arc_mode cos \
  --epcl_anchor fine_emotion \
  --cls_anchor default \
  --use_pcam \
  --pcam_heads 2 \
  --pcam_dropout 0.1 \
  --pcam_gate_bias -1.0 \
  --pcam_tau_attn 1.5 \
  --emo_bias_gate_init -2.0 \
  --use_pcgrad \
  --gate_warmup_steps 2500 \
  --mask_update_interval 1000 \
  --use_bias_annealing \
  --bias_anneal_start 2500 \
  --bias_anneal_steps 2500 \
  --bias_min_scale 0.05 \
  --use_emo_loss_ramp \
  --emo_loss_ramp_start 2500 \
  --emo_loss_ramp_steps 2500 \
  --emo_loss_ramp_max 1.25 \
  --emo_loss_ramp_shape bell \
  --use_composite_score \
  --composite_mode emo_loss \
  --composite_emo_loss_weight 5.0 \
  --min_save_step ${MIN_SAVE_STEP} \
  --patience ${PATIENCE} \
  ${EXTRA_FLAGS} \
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
  --gpu ${GPU_ID} \
  --use_arc_margin \
  ${EXTRA_FLAGS}

# 3. 汇总所有需要回传或归档的文件到统一输出目录
if [ -d "${HOME}/autodl-tmp" ]; then
    DOWNLOAD_DIR="${HOME}/autodl-tmp/download/${EXP_NAME}"
else
    DOWNLOAD_DIR="results/download/${EXP_NAME}"
fi
mkdir -p "${DOWNLOAD_DIR}"
cp -f "${LOG_FILE}" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "${OUTPUT_DIR}/eval_metrics.json" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "results/v9/${EXP_NAME}_eval.txt" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "results/v9/${EXP_NAME}_manifold.png" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "results/v9/${EXP_NAME}_results.txt" "${DOWNLOAD_DIR}/" 2>/dev/null || true

echo "======================================================================"
echo "[*] 全部完成！产出文件已汇总至:"
echo "    ${DOWNLOAD_DIR}/"
echo "    包含: ${EXP_NAME}.log, eval_metrics.json, *_eval.txt, *_manifold.png, *_results.txt"
echo "======================================================================"
ls -lh "${DOWNLOAD_DIR}/"

