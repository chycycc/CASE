#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 统一推断与学术指标评测脚本 (eval.sh)
# 一键度量 Test PPL, Greedy Dist-2, Sampling Unique, Alignment, DBI, 混淆矩阵
# ==============================================================================
set -e

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

# 待评测的黄金模型权重路径或实验目录 (默认优先检测 v9_trial10，可回退)
INPUT_TARGET=${1:-"save/v9_trial10"}
GPU_ID=${2:-"0"}
BATCH_SIZE=${3:-"64"}
EVAL_TARGET=${4:-"auto"}  # auto / composite / ppl

# 智能推断实验代号与路径
if [ -d "${INPUT_TARGET}" ] || [ -f "${INPUT_TARGET}" ]; then
    MODEL_PATH="${INPUT_TARGET}"
    EXP_NAME=$(basename "${INPUT_TARGET}" | sed 's/\.pth$//' | sed 's/\.pt$//')
elif [ -d "save/${INPUT_TARGET}" ]; then
    MODEL_PATH="save/${INPUT_TARGET}"
    EXP_NAME="${INPUT_TARGET}"
else
    # 回退到历史黄金基准
    MODEL_PATH="save/v9_trial9/CASE_7499_37.7145_composite_best"
    EXP_NAME="v9_trial9_benchmark"
fi

mkdir -p results/v9

# 依评测目标定制化输出报告文件名
if [ "${EVAL_TARGET}" = "composite" ]; then
    OUTPUT_TAG="${EXP_NAME}_composite"
elif [ "${EVAL_TARGET}" = "ppl" ]; then
    OUTPUT_TAG="${EXP_NAME}_ppl_best"
else
    OUTPUT_TAG="${EXP_NAME}"
fi

echo "======================================================================"
echo "[*] 启动 CASE-EPCL 全量自动化评估体系 (双检查点安全锁定模式)"
echo "    - 目标实验: ${EXP_NAME}"
echo "    - 权重路径: ${MODEL_PATH}"
echo "    - 评测偏好: ${EVAL_TARGET}"
echo "    - 执行设备: GPU ${GPU_ID} | 批次大小: ${BATCH_SIZE}"
echo "    - 报告产出: results/v9/${OUTPUT_TAG}_eval.txt"
echo "    - 流形产出: results/v9/${OUTPUT_TAG}_manifold.png"
echo "======================================================================"

# [修改项 5.1] 追加 "${@:5}" 透传额外命令行参数至 eval_pipeline.py
${pythonpath} src/scripts/eval_pipeline.py \
  --model_path "${MODEL_PATH}" \
  --eval_target "${EVAL_TARGET}" \
  --results_path "results/v9/${OUTPUT_TAG}_eval.txt" \
  --plot_path "results/v9/${OUTPUT_TAG}_manifold.png" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${GPU_ID}" \
  "${@:5}"


