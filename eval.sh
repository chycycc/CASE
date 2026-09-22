#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 统一推断与学术指标评测脚本 (eval.sh)
# 一键度量 Test PPL, Greedy Dist-2, Sampling Unique, Alignment, DBI, 混淆矩阵
# ==============================================================================
set -e

pythonpath='python'

# 待评测的黄金模型权重路径或实验目录 (默认优先检测 v9_trial2，可回退)
INPUT_TARGET=${1:-"save/v9_trial2"}
GPU_ID=${2:-"0"}
BATCH_SIZE=${3:-"64"}

# 智能推断实验代号与路径
if [ -d "${INPUT_TARGET}" ] || [ -f "${INPUT_TARGET}" ]; then
    MODEL_PATH="${INPUT_TARGET}"
    EXP_NAME=$(basename "${INPUT_TARGET}" | sed 's/\.pth$//' | sed 's/\.pt$//')
elif [ -d "save/${INPUT_TARGET}" ]; then
    MODEL_PATH="save/${INPUT_TARGET}"
    EXP_NAME="${INPUT_TARGET}"
else
    # 回退到历史黄金基准
    MODEL_PATH="save/epcl_v8_2_f/CASE_39999_37.0876"
    EXP_NAME="v8_2_f_benchmark"
fi

mkdir -p results/v9

echo "======================================================================"
echo "[*] 启动 CASE-EPCL 全量自动化评估体系 (独立隔离模式)"
echo "    - 目标实验: ${EXP_NAME}"
echo "    - 权重路径: ${MODEL_PATH}"
echo "    - 执行设备: GPU ${GPU_ID} | 批次大小: ${BATCH_SIZE}"
echo "    - 报告产出: results/v9/${EXP_NAME}_eval.txt"
echo "    - 流形产出: results/v9/${EXP_NAME}_manifold.png"
echo "======================================================================"

${pythonpath} src/scripts/eval_pipeline.py \
  --model_path "${MODEL_PATH}" \
  --results_path "results/v9/${EXP_NAME}_eval.txt" \
  --plot_path "results/v9/${EXP_NAME}_manifold.png" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${GPU_ID}"
