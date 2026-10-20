#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 Trial 12 10-Step Dry Run 验证脚本 (scratch/main_t12_dry.sh)
# 针对 RTX 3050Ti (4GB) 本地验证优化，仅跑 10 步验证全链路计算图与落盘
# ==============================================================================
set -e
set -o pipefail

# 智能匹配 Python 解释器路径
if [ -n "$PYTHON_BIN" ]; then
    pythonpath="$PYTHON_BIN"
elif [ -f "/e/Dev/condaData/envs_dirs/cem_env/python.exe" ]; then
    pythonpath="/e/Dev/condaData/envs_dirs/cem_env/python.exe"
elif [ -f "E:/Dev/condaData/envs_dirs/cem_env/python.exe" ]; then
    pythonpath="E:/Dev/condaData/envs_dirs/cem_env/python.exe"
else
    pythonpath="python"
fi

ENV_MODE=${1:-"4G"}
EXP_NAME=${2:-"v9_trial12_dry"}

DATASET='ED'
GPU_ID=${CUDA_VISIBLE_DEVICES:-"0"}
SEED=13
if [ "$EXP_NAME" = "v9_trial12_on" ] || [[ "$EXP_NAME" == *"_on"* ]]; then
    EXTRA_FLAGS="--use_proto_prefix --proto_prefix_len 4 --proto_prefix_init proto --proto_prefix_source ckpt --proto_prefix_prior_path results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best ${EXTRA_FLAGS:-}"
elif [ "$EXP_NAME" = "v9_trial12_rand" ] || [[ "$EXP_NAME" == *"_rand"* ]]; then
    EXTRA_FLAGS="--use_proto_prefix --proto_prefix_len 4 --proto_prefix_init random --proto_prefix_init_npz save/v9_trial12_on/proto_prefix_init.npz ${EXTRA_FLAGS:-}"
else
    EXTRA_FLAGS="${EXTRA_FLAGS:-}"
fi

if [ "$ENV_MODE" = "4G" ]; then
    BATCH_SIZE=4
    ACCUM_STEPS=8
    PRECISION="fp32"
    LR=0.000576
    WARMUP=16
    CHECK_ITER=5
    MAX_STEP=10
    MIN_SAVE_STEP=1
    PATIENCE=10
    
    LR_DECAY_START=10
    LR_DECAY_STEPS=10
    GATE_WARMUP=10
    MASK_INTERVAL=5
    BIAS_ANNEAL_START=10
    BIAS_ANNEAL_STEPS=10
    EMO_RAMP_START=10
    EMO_RAMP_STEPS=10
    
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"
else
    BATCH_SIZE=32
    ACCUM_STEPS=1
    PRECISION="fp32"
    LR=0.000576
    WARMUP=10
    CHECK_ITER=5
    MAX_STEP=10
    MIN_SAVE_STEP=1
    PATIENCE=10
    
    LR_DECAY_START=10
    LR_DECAY_STEPS=10
    GATE_WARMUP=10
    MASK_INTERVAL=5
    BIAS_ANNEAL_START=10
    BIAS_ANNEAL_STEPS=10
    EMO_RAMP_START=10
    EMO_RAMP_STEPS=10
    
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"
fi

mkdir -p logs
mkdir -p results/v9
mkdir -p "${OUTPUT_DIR}"

echo "======================================================================"
echo "[*] 启动 T12 Dry Run (10 Steps):"
echo "    - 实验代号: ${EXP_NAME}"
echo "    - 硬件模式: ${ENV_MODE} (Batch: ${BATCH_SIZE}, Accum: ${ACCUM_STEPS})"
echo "    - 额外标志: ${EXTRA_FLAGS:-'(无)'}"
echo "    - 权重目录: ${OUTPUT_DIR}"
echo "    - 日志文件: ${LOG_FILE}"
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
  --lr_decay_start_step ${LR_DECAY_START} \
  --lr_decay_steps ${LR_DECAY_STEPS} \
  --check_iter ${CHECK_ITER} \
  --max_step ${MAX_STEP} \
  --pretrain_epoch 0 \
  --skip_test \
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
  --gate_warmup_steps ${GATE_WARMUP} \
  --mask_update_interval ${MASK_INTERVAL} \
  --use_bias_annealing \
  --bias_anneal_start ${BIAS_ANNEAL_START} \
  --bias_anneal_steps ${BIAS_ANNEAL_STEPS} \
  --bias_min_scale 0.05 \
  --use_emo_loss_ramp \
  --emo_loss_ramp_start ${EMO_RAMP_START} \
  --emo_loss_ramp_steps ${EMO_RAMP_STEPS} \
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
echo "[*] 训练 dry run 完成，启动评测链路自适应抽样测试..."
echo "======================================================================"

${pythonpath} src/scripts/eval_pipeline.py \
  --model_path ${OUTPUT_DIR} \
  --results_path "results/v9/${EXP_NAME}_eval.txt" \
  --plot_path "results/v9/${EXP_NAME}_manifold.png" \
  --batch_size 1 \
  --gen_samples 1 \
  --skip_manifold \
  --skip_test_eval \
  --gpu ${GPU_ID} \
  --use_arc_margin \
  ${EXTRA_FLAGS} 2>&1 | tee -a ${LOG_FILE}

echo "======================================================================"
echo "[*] Dry run 全链路验证完成: ${LOG_FILE}"
echo "======================================================================"
