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
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 4 梯度累加 8 次 (等效 Batch 32) + 显存峰值 < 900MB"
    echo "    - 时序映射: 调度按 8x 映射；梯度累加仅作工程烟测，不保证云端优化轨迹等价"
    echo "======================================================================"
    BATCH_SIZE=4
    ACCUM_STEPS=8
    PRECISION="fp32"
    LR=0.000576
    WARMUP=16000           # 对应 2000 次更新预热
    CHECK_ITER=4000        # 对应每 500 次更新评估一次
    MAX_STEP=80000         # 对应 10000 次更新上限
    MIN_SAVE_STEP=24000    # 对应 3000 次更新保护期
    PATIENCE=6
    
    # 4G 调度按 8x 映射；不据此宣称累加优化路径与 24G 数学等价。
    LR_DECAY_START=24000   # 3000 * 8
    LR_DECAY_STEPS=48000   # 6000 * 8
    GATE_WARMUP=20000      # 2500 * 8
    MASK_INTERVAL=8000     # 1000 * 8
    BIAS_ANNEAL_START=20000
    BIAS_ANNEAL_STEPS=20000
    EMO_RAMP_START=20000
    EMO_RAMP_STEPS=20000
    
    EXTRA_FLAGS=${EXTRA_FLAGS:-""}
    OUTPUT_DIR="save/${EXP_NAME}/"
    LOG_FILE="logs/${EXP_NAME}.log"
else
    echo "======================================================================"
    echo "[*] 加载 24GB 显存生产级全量模式 (RTX 4090 / 3090 旗舰训练)"
    echo "    - 实验代号: ${EXP_NAME}"
    echo "    - 特性: FP32 全精度原生无损 + 单步 Batch 32 零累加 (更新步数 10,000 次) + 显存安全可控"
    echo "======================================================================"
    BATCH_SIZE=32
    ACCUM_STEPS=1
    PRECISION="fp32"
    LR=0.000576
    WARMUP=2000
    CHECK_ITER=500
    MAX_STEP=10000
    MIN_SAVE_STEP=3000
    PATIENCE=6
    
    LR_DECAY_START=3000
    LR_DECAY_STEPS=6000
    GATE_WARMUP=2500
    MASK_INTERVAL=1000
    BIAS_ANNEAL_START=2500
    BIAS_ANNEAL_STEPS=2500
    EMO_RAMP_START=2500
    EMO_RAMP_STEPS=2500
    
    EXTRA_FLAGS=${EXTRA_FLAGS:-""}
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

# Trial12 gate variants use this same entry point; no extra launcher.
case "$EXP_NAME" in
    v9_trial12_gate_proto|v9_trial12_gate_rand|v9_trial12_gate_norm_proto|v9_trial12_gate_norm_rand)
        if [ -n "$EXTRA_FLAGS" ]; then
            echo "[T12] This named profile fixes EXTRA_FLAGS; unset inherited overrides first." >&2
            exit 1
        fi
        if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ]; then
            echo "[T12] Output already exists: $EXP_NAME; refusing to overwrite." >&2
            exit 1
        fi
        T12_MODE=proto
        case "$EXP_NAME" in *_rand) T12_MODE=random ;; esac
        T12_SOURCE=artifact
        T12_NORM_FLAGS=""
        T12_DEFAULT_INIT="save/v9_trial12/proto_prefix_init.npz"
        case "$EXP_NAME" in
            v9_trial12_gate_norm_*)
                T12_SOURCE=frozen
                T12_NORM_FLAGS="--use_proto_prefix_norm --prefix_diagnostics"
                T12_PARENT=v9_trial12_gate_proto
                [ "$T12_MODE" = "random" ] && T12_PARENT=v9_trial12_gate_rand
                T12_DEFAULT_INIT="save/${T12_PARENT}/proto_prefix_init.npz"
                if [ ! -f "$T12_DEFAULT_INIT" ]; then
                    T12_DEFAULT_INIT="results/cloud_archive/${T12_PARENT}/proto_prefix_init.npz"
                fi
                ;;
        esac
        T12_INIT="${T12_INIT_NPZ:-$T12_DEFAULT_INIT}"
        "$pythonpath" - "$T12_INIT" "$T12_SOURCE" "$T12_MODE" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    raise FileNotFoundError("Archived T12 initializer required: " + str(p))
if sys.argv[2] == "frozen":
    import numpy as np
    # 预检只导入numpy；导入训练模块会额外加载CUDA动态库。
    expected = {
        "proto": "f061c61c9499c202e82586018cf6755364683c434ca013eafe6ec8b20d414169",
        "random": "c079bcb310653061f3c704140eda6d367dffdaf3d309178066b571a355c0de94",
    }
    with np.load(str(p), allow_pickle=False) as data:
        init = data["init"].astype("float32")
    if init.shape != (4, 300) or hashlib.sha256(init.tobytes()).hexdigest() != expected[sys.argv[3]]:
        raise RuntimeError("Actual initializer mismatch; do not regenerate either group")
elif hashlib.sha256(p.read_bytes()).hexdigest() != "341ce94027ce7af25bdb55f2f14cd2b8353098932f0ced76136575d9d908f898":
    raise RuntimeError("T12 initializer hash mismatch; do not recompute KMeans")
print("[T12] Frozen initializer verified:", p, "mode=" + sys.argv[3], "source=" + sys.argv[2])
PY
        T12_ARC_RAMP=2500
        [ "$ENV_MODE" = "4G" ] && T12_ARC_RAMP=20000
        EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps $T12_ARC_RAMP --use_proto_prefix --proto_prefix_len 4 --use_proto_prefix_gate --proto_prefix_gate_init 0.0 --proto_prefix_init $T12_MODE --proto_prefix_source $T12_SOURCE --proto_prefix_init_npz $T12_INIT $T12_NORM_FLAGS"
        ;;
    v10_trial1_sep_dec_head)
        if [ -n "$EXTRA_FLAGS" ]; then
            echo "[V10 T1] This named profile fixes EXTRA_FLAGS; unset inherited overrides first." >&2
            exit 1
        fi
        if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ]; then
            echo "[V10 T1] Output already exists: $EXP_NAME; refusing to overwrite." >&2
            exit 1
        fi
        V10_ARC_RAMP=2500
        [ "$ENV_MODE" = "4G" ] && V10_ARC_RAMP=20000
        EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps $V10_ARC_RAMP --use_separate_dec_emo_head"
        ;;
esac

# 2. 建立独立日志与实验产出目录（按实验名前缀自适应 results/v10 与 results/v9）
RESULTS_DIR="results/v9"
case "$EXP_NAME" in
    v10_*) RESULTS_DIR="results/v10" ;;
esac
mkdir -p logs
mkdir -p "${RESULTS_DIR}"

echo "[*] 启动配置全览:"
echo "    - 实验代号: ${EXP_NAME}"
echo "    - 硬件模式: ${ENV_MODE} (Batch: ${BATCH_SIZE}, Accum: ${ACCUM_STEPS}, Precision: ${PRECISION})"
echo "    - 注入参数: ${EXTRA_FLAGS:-'(无)'}"
echo "    - 权重保存: ${OUTPUT_DIR}"
echo "    - 训练日志: ${LOG_FILE}"
echo "    - 评估间隔: 每 ${CHECK_ITER} 步 | 最大步数: ${MAX_STEP} 步 | 保护期: ${MIN_SAVE_STEP} 步"
echo "======================================================================"

# BEGIN_RUNTIME_METADATA
# 独立进程探测真实默认清零行为，不改变训练进程的 RNG 或优化器。
record_runtime_metadata() {
"${pythonpath}" - "${OUTPUT_DIR}/env_metadata.json" "${ENV_MODE}" "${EXP_NAME}" <<'PY'
import inspect, json, os, platform, sys
import numpy, sklearn, torch

def probe_zero_grad(device):
    p = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.Adam([p])
    p.grad = torch.ones_like(p)
    opt.zero_grad()
    return {"device": device, "signature": str(inspect.signature(opt.zero_grad)),
            "grad_is_none": p.grad is None,
            "grad_is_zero_tensor": p.grad is not None and bool((p.grad == 0).all().item())}

def validate_zero_grad(info, mode, experiment):
    # V10 正式路径要求分类头不因默认清零语义进入 PCGrad shared_params。
    # 不改 zero_grad 参数；运行时不符时在训练前终止，保留元数据供定位。
    return not (mode == "24G" and experiment == "v10_trial1_sep_dec_head"
                and any(not p["grad_is_none"] for p in info["optimizer_zero_grad"]))

info = {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hardware_mode": sys.argv[2], "experiment": sys.argv[3], "recorded_before_training": True,
        "optimizer_zero_grad": [probe_zero_grad("cpu")]}
if torch.cuda.is_available():
    info["optimizer_zero_grad"].append(probe_zero_grad("cuda:0"))
info["runtime_contract_passed"] = validate_zero_grad(info, sys.argv[2], sys.argv[3])
os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
with open(sys.argv[1] + ".tmp", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
print("[Runtime Preflight] " + json.dumps(info, ensure_ascii=False), flush=True)
if not info["runtime_contract_passed"]:
    raise RuntimeError("V10 24G requires default Adam.zero_grad() to set existing gradients to None; "
                       "runtime mismatch, training was not started. See env_metadata.json.")
PY
}
# END_RUNTIME_METADATA

record_runtime_metadata 2>&1 | tee "${LOG_FILE}"

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
  --model_file_path ${OUTPUT_DIR} 2>&1 | tee -a ${LOG_FILE}

echo "======================================================================"
echo "[*] 训练完成！自动启动全量学术指标自动化审计..."
echo "======================================================================"

${pythonpath} src/scripts/eval_pipeline.py \
  --model_path ${OUTPUT_DIR} \
  --results_path "${RESULTS_DIR}/${EXP_NAME}_eval.txt" \
  --plot_path "${RESULTS_DIR}/${EXP_NAME}_manifold.png" \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU_ID} \
  --use_arc_margin \
  ${EXTRA_FLAGS} 2>&1 | tee -a ${LOG_FILE}

# BEGIN_ARTIFACT_PACKAGING
# 与本地验收共用真实打包函数；不依赖前缀初始化文件存在。
package_artifacts() {
mkdir -p "${DOWNLOAD_DIR}"
if [ "${EXP_NAME}" = "v10_trial1_sep_dec_head" ]; then
    for REQUIRED_FILE in env_metadata.json dec_emo_diagnostics.jsonl; do
        if [ ! -s "${OUTPUT_DIR}/${REQUIRED_FILE}" ]; then
            echo "[V10 T1] Missing required artifact: ${OUTPUT_DIR}/${REQUIRED_FILE}" >&2
            return 1
        fi
    done
    if ! compgen -G "${OUTPUT_DIR}/CASE_*_composite_best" > /dev/null || ! compgen -G "${OUTPUT_DIR}/CASE_*_ppl_best" > /dev/null; then
        echo "[V10 T1] Both composite_best and ppl_best checkpoints are required." >&2
        return 1
    fi
fi
cp -f "${LOG_FILE}" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "${OUTPUT_DIR}/eval_metrics.json" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "${RESULTS_DIR}/${EXP_NAME}_eval.txt" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "${RESULTS_DIR}/${EXP_NAME}_manifold.png" "${DOWNLOAD_DIR}/" 2>/dev/null || true
cp -f "${RESULTS_DIR}/${EXP_NAME}_results.txt" "${DOWNLOAD_DIR}/" 2>/dev/null || true
for INIT_FILE in proto_prefix_init.npz prefix_init_metadata.json prefix_diagnostics.jsonl dec_emo_diagnostics.jsonl env_metadata.json; do
    if [ -f "${OUTPUT_DIR}/${INIT_FILE}" ]; then
        cp -f "${OUTPUT_DIR}/${INIT_FILE}" "${DOWNLOAD_DIR}/"
    fi
done
if compgen -G "${OUTPUT_DIR}/CASE_*_best" > /dev/null; then
    sha256sum "${OUTPUT_DIR}"/CASE_*_best > "${DOWNLOAD_DIR}/checkpoint_sha256.txt"
fi
sha256sum main.sh main.py src/models/common.py src/models/CASE/model.py src/utils/config.py src/utils/prefix_runtime.py src/utils/data/loader.py src/scripts/eval_pipeline.py src/scripts/eval_prototype_ncm.py > "${DOWNLOAD_DIR}/source_sha256.txt"
}
# END_ARTIFACT_PACKAGING

# 3. 汇总所有需要回传或归档的文件到统一输出目录
if [ -d "${HOME}/autodl-tmp" ]; then
    DOWNLOAD_DIR="${HOME}/autodl-tmp/download/${EXP_NAME}"
else
    DOWNLOAD_DIR="results/download/${EXP_NAME}"
fi
package_artifacts

echo "======================================================================"
echo "[*] 全部完成！产出文件已汇总至:"
echo "    ${DOWNLOAD_DIR}/"
echo "    包含: ${EXP_NAME}.log, eval_metrics.json, *_eval.txt, *_manifold.png, *_results.txt"
echo "======================================================================"
ls -lh "${DOWNLOAD_DIR}/"

