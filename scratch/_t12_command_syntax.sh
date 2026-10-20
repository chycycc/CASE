PRIOR="results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best"
EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps 2500 --use_proto_prefix --proto_prefix_len 4 --proto_prefix_init proto --proto_prefix_source ckpt --proto_prefix_prior_path $PRIOR" \
  bash main.sh 24G v9_trial12

EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps 2500 --use_proto_prefix --proto_prefix_len 4 --proto_prefix_init random --proto_prefix_init_npz save/v9_trial12/proto_prefix_init.npz" \
  bash main.sh 24G v9_trial12_C

T12_PYTHON="E:/Dev/condaData/envs_dirs/cem_env/python.exe"
export NLTK_DATA="C:/Users/chy/AppData/Roaming/nltk_data"
"$T12_PYTHON" -X utf8 tests/test_trial12_prefix.py -v
"$T12_PYTHON" -X utf8 tests/test_trial12_scope.py -v
"$T12_PYTHON" -X utf8 src/scripts/audit_trial12_dry_run.py --output results/v9/trial12_handoff_audit/dry_run.json
"$T12_PYTHON" -X utf8 scratch/_t12_xavier_check.py
"$T12_PYTHON" -X utf8 scratch/_t12_ncm_smoke.py save/v9_trial12_on/CASE_9_21624.4067_composite_best results/v9/trial12_handoff_audit/ncm_smoke_one_sample.json

PRIOR="results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best"
test -f "$PRIOR" || exit 1
sha256sum "$PRIOR"
python -u -c "from src.utils.config import config; from src.models.CASE.model import CASE; p=CASE._load_proto_from_ckpt(None); e=CASE._deterministic_project(None,p,4); print(tuple(e.shape))" --proto_prefix_prior_path "$PRIOR"

python -u src/scripts/eval_prototype_ncm.py --model_path save/v9_trial12 --batch_size 32 --output_dir save/diag_trial12
python -u src/scripts/eval_prototype_ncm.py --model_path save/v9_trial12_C --batch_size 32 --output_dir save/diag_trial12_C
