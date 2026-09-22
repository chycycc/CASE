#!/bin/bash
# ==============================================================================
# CASE-EPCL V9 统一推断与学术指标评测脚本 (eval.sh)
# 一键度量 Test PPL, Greedy Dist-2, Sampling Unique, Alignment, DBI, 混淆矩阵
# ==============================================================================
set -e

pythonpath='python'

# 待评测的黄金模型权重路径 (支持传入具体权重文件或保存目录，默认扫描 save/v9_24g_baseline)
MODEL_PATH=${1:-"save/v9_24g_baseline"}
GPU_ID=${2:-"0"}
BATCH_SIZE=${3:-"32"}

echo "======================================================================"
echo "[*] 启动 CASE-EPCL 全量自动化评估体系"
echo "    - 评估权重: ${MODEL_PATH}"
echo "    - 执行设备: GPU ${GPU_ID} | 批次大小: ${BATCH_SIZE}"
echo "======================================================================"

${pythonpath} src/scripts/eval_pipeline.py \
  --model_path ${MODEL_PATH} \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU_ID}
