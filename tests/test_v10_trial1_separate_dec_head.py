# -*- coding: utf-8 -*-
"""V10 Trial 1 回归测试：主情绪分类头与解码端辅助分类头解耦、RNG隔离、梯度恒等式与检查点审计。"""
import contextlib
import ast
import copy
import hashlib
import io
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

with contextlib.redirect_stdout(io.StringIO()):
    from src.utils.config import config
    from src.models.common import evaluate, count_parameters
    from src.utils.prefix_runtime import (
        DEC_EMO_HEAD_MODE,
        checkpoint_separate_dec_emo_head,
        compute_dec_emo_head_divergence,
        capture_rng_state,
        assert_rng_equal,
        isolated_rng,
    )


class V10Trial1UnitTests(unittest.TestCase):
    """验证 V10 Trial 1 的参数隔离、梯度恒等式、检查点校验与 Valid 诊断。"""

    def setUp(self):
        torch.manual_seed(13)
        np.random.seed(13)
        random.seed(13)
        torch.set_num_threads(1)

    def test_01_default_config_and_isolated_rng_instantiation(self):
        """测试开关默认为 False，且在 isolated_rng 下实例化 Linear 不改变四路随机状态。"""
        self.assertFalse(getattr(config, "use_separate_dec_emo_head", False))
        before = capture_rng_state()
        with isolated_rng():
            head = nn.Linear(300, 32)
        after = capture_rng_state()
        assert_rng_equal(before, after)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 9632)

    def test_02_checkpoint_metadata_and_topology_validation(self):
        """测试检查点元数据与权重键的交叉校验及异常熔断（铁律 IV）。"""
        # 1. 历史旧检查点（无 dec_emo_head_mode，无 dec_emotion_linear 键）-> False
        legacy_ckpt = {
            "model": {
                "emotion_linear.weight": torch.zeros(32, 300),
                "emotion_linear.bias": torch.zeros(32),
            }
        }
        self.assertFalse(checkpoint_separate_dec_emo_head(legacy_ckpt))

        # 2. 显式 shared 模式且无独立头键 -> False
        shared_ckpt = {
            "dec_emo_head_mode": "shared",
            "model": dict(legacy_ckpt["model"]),
        }
        self.assertFalse(checkpoint_separate_dec_emo_head(shared_ckpt))

        # 3. 显式 separate_linear_v1 模式且含完整 weight/bias -> True
        sep_ckpt = {
            "dec_emo_head_mode": DEC_EMO_HEAD_MODE,
            "model": {
                "emotion_linear.weight": torch.zeros(32, 300),
                "emotion_linear.bias": torch.zeros(32),
                "dec_emotion_linear.weight": torch.ones(32, 300),
                "dec_emotion_linear.bias": torch.ones(32),
            },
        }
        self.assertTrue(checkpoint_separate_dec_emo_head(sep_ckpt))

        # 4. 异常情况必须全部抛出 RuntimeError
        # 4a. 未知的 dec_emo_head_mode
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head({"dec_emo_head_mode": "unknown_mode", "model": {}})

        # 4b. 声明 separate_linear_v1 但缺少权重键
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head({"dec_emo_head_mode": DEC_EMO_HEAD_MODE, "model": dict(legacy_ckpt["model"])})

        # 4c. 声明 shared 但包含 dec_emotion_linear 权重键
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head({"dec_emo_head_mode": "shared", "model": dict(sep_ckpt["model"])})

        # 4d. 仅有 weight 缺少 bias
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head({
                "dec_emo_head_mode": DEC_EMO_HEAD_MODE,
                "model": {
                    "emotion_linear.weight": torch.zeros(32, 300),
                    "emotion_linear.bias": torch.zeros(32),
                    "dec_emotion_linear.weight": torch.ones(32, 300),
                },
            })

        # 4e. 形状与主分类头不匹配
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head({
                "dec_emo_head_mode": DEC_EMO_HEAD_MODE,
                "model": {
                    "emotion_linear.weight": torch.zeros(32, 300),
                    "emotion_linear.bias": torch.zeros(32),
                    "dec_emotion_linear.weight": torch.ones(16, 300),
                    "dec_emotion_linear.bias": torch.ones(16),
                },
            })

        # 4f. 含有 NaN/Inf
        bad_nan = copy.deepcopy(sep_ckpt)
        bad_nan["model"]["dec_emotion_linear.weight"][0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            checkpoint_separate_dec_emo_head(bad_nan)

    def test_03_head_divergence_computation(self):
        """测试共享模式与独立模式下两分类头余弦与距离计算。"""
        holder = SimpleNamespace(
            use_separate_dec_emo_head=False,
            emotion_linear=nn.Linear(12, 4),
            dec_emotion_linear=None,
        )
        div_shared = compute_dec_emo_head_divergence(holder)
        self.assertEqual(div_shared["head_mode"], "shared")
        self.assertEqual(div_shared["row_cosine_mean"], 1.0)
        self.assertEqual(div_shared["weight_max_abs_diff"], 0.0)

        holder.use_separate_dec_emo_head = True
        holder.dec_emotion_linear = copy.deepcopy(holder.emotion_linear)
        div_same = compute_dec_emo_head_divergence(holder)
        self.assertEqual(div_same["head_mode"], DEC_EMO_HEAD_MODE)
        self.assertAlmostEqual(div_same["row_cosine_mean"], 1.0, places=12)
        self.assertEqual(div_same["weight_max_abs_diff"], 0.0)
        self.assertEqual(div_same["bias_l2_distance"], 0.0)

        with torch.no_grad():
            holder.dec_emotion_linear.weight.mul_(-1.0)
            holder.dec_emotion_linear.bias.add_(1.0)
        div_opp = compute_dec_emo_head_divergence(holder)
        self.assertAlmostEqual(div_opp["row_cosine_mean"], -1.0, places=6)
        self.assertAlmostEqual(div_opp["bias_l2_distance"], 2.0, places=6)

    def test_04_accum1_pcgrad_and_pre_clip_norm_identities(self):
        """验证 accum_steps=1 下 PCGrad 共享参数隔离、梯度加和与裁剪前全局范数平方恒等式。"""
        torch.manual_seed(2026)
        enc_shared = nn.Linear(16, 12)
        dec_shared = nn.Linear(12, 12)
        gen_head = nn.Linear(12, 20)
        emo_head = nn.Linear(12, 6)

        x = torch.randn(4, 16)
        y_gen = torch.randint(0, 20, (4,))
        y_emo = torch.randint(0, 6, (4,))

        def run_step(separate: bool):
            m_enc = copy.deepcopy(enc_shared)
            m_dec = copy.deepcopy(dec_shared)
            m_gen = copy.deepcopy(gen_head)
            m_emo = copy.deepcopy(emo_head)
            m_dec_emo = copy.deepcopy(emo_head) if separate else None
            params = list(m_enc.parameters()) + list(m_dec.parameters()) + list(m_gen.parameters()) + list(m_emo.parameters())
            if m_dec_emo is not None:
                params += list(m_dec_emo.parameters())

            for p in params:
                p.grad = None

            h_enc = torch.tanh(m_enc(x))
            h_dec = torch.tanh(m_dec(h_enc))
            loss_gen = F.cross_entropy(m_gen(h_dec), y_gen)
            loss_emo_main = F.cross_entropy(m_emo(h_enc), y_emo)
            aux_head = m_dec_emo if separate else m_emo
            loss_emo_dec = F.cross_entropy(aux_head(h_dec), y_emo)
            loss_emo = loss_emo_main + 0.1 * loss_emo_dec

            # 模拟 CASE.train_one_batch 在 accum_steps=1 下的 PCGrad 流程
            loss_gen.backward(retain_graph=True)
            grads_gen = {p: p.grad.clone() for p in params if p.grad is not None}
            for p in params:
                p.grad = None
            loss_emo.backward()
            grads_emo = {p: p.grad.clone() for p in params if p.grad is not None}

            shared_params = [p for p in grads_gen if p in grads_emo]
            shared_set = set(shared_params)
            # 关键断言：分类头绝不进入 grads_gen 或 shared_params
            for p in m_emo.parameters():
                self.assertNotIn(p, grads_gen)
                self.assertNotIn(p, shared_set)
            if m_dec_emo is not None:
                for p in m_dec_emo.parameters():
                    self.assertNotIn(p, grads_gen)
                    self.assertNotIn(p, shared_set)

            if shared_params:
                dot_prod = sum(torch.sum(grads_gen[p] * grads_emo[p]) for p in shared_params)
                if dot_prod < 0:
                    norm_gen_sq = sum(torch.sum(grads_gen[p] * grads_gen[p]) for p in shared_params)
                    norm_emo_sq = sum(torch.sum(grads_emo[p] * grads_emo[p]) for p in shared_params)
                    coeff_emo = dot_prod / (norm_gen_sq + 1e-8)
                    coeff_gen = dot_prod / (norm_emo_sq + 1e-8)
                    for p in shared_params:
                        proj_emo = grads_emo[p] - coeff_emo * grads_gen[p]
                        proj_gen = grads_gen[p] - coeff_gen * grads_emo[p]
                        grads_emo[p] = proj_emo
                        grads_gen[p] = proj_gen

            for p in set(grads_gen.keys()).union(set(grads_emo.keys())):
                g_gen = grads_gen.get(p, None)
                g_emo = grads_emo.get(p, None)
                if g_gen is not None and g_emo is not None:
                    p.grad = g_gen + g_emo
                elif g_gen is not None:
                    p.grad = g_gen
                elif g_emo is not None:
                    p.grad = g_emo

            total_norm_sq = sum(torch.sum(p.grad.double() ** 2).item() for p in params if p.grad is not None)
            return {
                "losses": (loss_gen.item(), loss_emo_main.item(), loss_emo_dec.item()),
                "common_grads": [p.grad.clone() for p in (list(m_enc.parameters()) + list(m_dec.parameters()) + list(m_gen.parameters()))],
                "emo_grads": (m_emo.weight.grad.clone(), m_emo.bias.grad.clone()),
                "dec_emo_grads": (m_dec_emo.weight.grad.clone(), m_dec_emo.bias.grad.clone()) if m_dec_emo is not None else None,
                "total_norm_sq": total_norm_sq,
            }

        res_off = run_step(False)
        res_on = run_step(True)

        # 1. 前向损失逐位相等
        self.assertEqual(res_off["losses"], res_on["losses"])
        # 2. 公共非分类头梯度逐位相等
        for g_a, g_b in zip(res_off["common_grads"], res_on["common_grads"]):
            self.assertTrue(torch.equal(g_a, g_b))
        # 3. 分类头梯度加和恒等式 g_shared = g_enc + g_dec
        gw_shared, gb_shared = res_off["emo_grads"]
        gw_enc, gb_enc = res_on["emo_grads"]
        gw_dec, gb_dec = res_on["dec_emo_grads"]
        self.assertLessEqual((gw_shared - (gw_enc + gw_dec)).abs().max().item(), 1e-6)
        self.assertLessEqual((gb_shared - (gb_enc + gb_dec)).abs().max().item(), 1e-6)
        # 4. 裁剪前总范数平方恒等式 N_split^2 - N_shared^2 = -2 <g_enc, g_dec>
        lhs = res_on["total_norm_sq"] - res_off["total_norm_sq"]
        inner = (torch.sum(gw_enc.double() * gw_dec.double()) + torch.sum(gb_enc.double() * gb_dec.double())).item()
        rhs = -2.0 * inner
        self.assertLessEqual(abs(lhs - rhs) / max(1.0, res_off["total_norm_sq"]), 1e-6)

    def test_05_valid_evaluate_records_sample_mean_and_preserves_rng(self):
        """验证 evaluate(ty='valid') 在非整除短尾批次下准确记录样本等权 CE、样本 micro Acc 且不消耗 RNG。"""
        class DummyCASE(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab = SimpleNamespace()
                self.epoch = 499
                self.use_separate_dec_emo_head = True
                self.emotion_linear = nn.Linear(8, 4)
                self.dec_emotion_linear = copy.deepcopy(self.emotion_linear)
                with torch.no_grad():
                    self.dec_emotion_linear.weight.add_(0.25)
                self._last_eval_dec_emo_stats = None

            def train_one_batch(self, batch, iter, train=False):
                bsz = batch["bsz"]
                ce_sum = batch["ce_sum"]
                correct = batch["correct"]
                self._last_eval_dec_emo_stats = {
                    "batch_ce": ce_sum / bsz,
                    "batch_size": bsz,
                    "ce_sum": ce_sum,
                    "correct": correct,
                }
                return (1.0, 0.1, 0.2, 3.5, float(np.exp(3.5)), 0.0, 0.0, 2.3, 0.4, 0.05, 0.0)

        # 构造 3 个批次：batch_size分别为 4, 4, 2（总样本 N=10，模拟 Valid 180 批末批短尾）
        batches = [
            {"bsz": 4, "ce_sum": 8.0, "correct": 2},   # batch_ce = 2.0
            {"bsz": 4, "ce_sum": 12.0, "correct": 1},  # batch_ce = 3.0
            {"bsz": 2, "ce_sum": 10.0, "correct": 1},  # batch_ce = 5.0
        ]
        # 样本等权均值 = (8 + 12 + 10) / 10 = 3.0；batch macro = (2 + 3 + 5) / 3 = 3.333333...
        # 样本 micro Acc = (2 + 1 + 1) / 10 = 0.4
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as tmp:
            dummy = DummyCASE()
            before = capture_rng_state()
            with patch.multiple(config, model="case", dataset="ED", save_path=tmp), \
                 patch("src.models.common.Translator", return_value=object()), \
                 contextlib.redirect_stdout(io.StringIO()):
                out = evaluate(dummy, batches, ty="valid", max_dec_step=30)
            assert_rng_equal(before, capture_rng_state())
            self.assertEqual(len(out), 10)
            diag_file = Path(tmp) / "dec_emo_diagnostics.jsonl"
            self.assertTrue(diag_file.is_file())
            record = json.loads(diag_file.read_text(encoding="utf-8").strip())
            self.assertEqual(record["step"], 499)
            self.assertEqual(record["scope"]["evaluated_n"], 10)
            self.assertEqual(record["scope"]["num_batches"], 3)
            self.assertAlmostEqual(record["dec_emo_ce_sample_mean"], 3.0, places=12)
            self.assertAlmostEqual(record["dec_emo_ce_batch_macro"], 10.0 / 3.0, places=12)
            self.assertAlmostEqual(record["dec_emo_acc_sample_micro"], 0.4, places=12)
            self.assertEqual(record["dec_emo_correct"], 4)
            self.assertEqual(record["head_divergence"]["head_mode"], DEC_EMO_HEAD_MODE)
            self.assertLess(record["head_divergence"]["row_cosine_mean"], 1.0)

    def test_06_eval_loaders_detect_and_enforce_separate_dec_head_audit(self):
        """验证 eval_pipeline 与 eval_prototype_ncm 识别独立分类头并在缺失/多余时触发熔断（铁律 IV）。"""
        from tests.test_trial12_prefix import source_function
        from src.utils.prefix_runtime import checkpoint_prefix_norm

        def run_loader(file, fn, separate=False, dec_mode=None, missing=(), unexpected=()):
            cfg = SimpleNamespace(device="cpu", use_proto_prefix=False, use_separate_dec_emo_head=not separate)
            state = {
                "model": {
                    "epcl_criterion.prototypes": torch.zeros(32, 300),
                    "emotion_linear.weight": torch.zeros(32, 300),
                    "emotion_linear.bias": torch.zeros(32),
                }
            }
            if separate:
                state["model"]["dec_emotion_linear.weight"] = torch.zeros(32, 300)
                state["model"]["dec_emotion_linear.bias"] = torch.zeros(32)
            if dec_mode is not None:
                state["dec_emo_head_mode"] = dec_mode

            class MockModel:
                epcl_criterion = object()
                def to(self, device): return self
                def eval(self): pass
                def print_proto_prefix_gate_status(self, **kwargs): pass
                def print_dec_emo_head_status(self, **kwargs): pass
                def load_state_dict(self, sd, strict):
                    return list(missing), list(unexpected)

            namespace = dict(
                os=os, config=cfg, torch=torch, CASE=lambda *a, **kw: MockModel(),
                checkpoint_prefix_norm=checkpoint_prefix_norm,
                PROJECT_ROOT=str(ROOT), set_seed=lambda: None,
                prepare_data_seq=lambda **kw: (None, None, None, None, 32, 8),
            )
            loader = source_function(file, fn, namespace)
            with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
                checkpoint_path = Path(directory) / "checkpoint.pt"
                checkpoint_path.touch()
                with patch.object(torch, "load", return_value=state), contextlib.redirect_stdout(io.StringIO()):
                    loader(str(checkpoint_path))
            return cfg

        for file, fn in [("src/scripts/eval_pipeline.py", "load_v8_model"),
                         ("src/scripts/eval_prototype_ncm.py", "safe_load_checkpoint")]:
            # 1. 独立头检查点自动开启 config.use_separate_dec_emo_head
            cfg_on = run_loader(file, fn, separate=True, dec_mode=DEC_EMO_HEAD_MODE)
            self.assertTrue(cfg_on.use_separate_dec_emo_head)
            # 2. 共享头检查点自动重置 config.use_separate_dec_emo_head = False
            cfg_off = run_loader(file, fn, separate=False, dec_mode="shared")
            self.assertFalse(cfg_off.use_separate_dec_emo_head)
            # 3. 独立头模式下缺失 dec_emotion_linear.weight 必须报错
            with self.assertRaises((RuntimeError, AssertionError)):
                run_loader(file, fn, separate=True, dec_mode=DEC_EMO_HEAD_MODE,
                           missing=["dec_emotion_linear.weight"])
            # 4. 共享头模式下 unexpected 含 dec_emotion_linear.weight 必须报错
            with self.assertRaises((RuntimeError, AssertionError)):
                run_loader(file, fn, separate=False, dec_mode="shared",
                           unexpected=["dec_emotion_linear.weight"])
            # 5. 元数据与权重矛盾必须报错
            with self.assertRaises(RuntimeError):
                run_loader(file, fn, separate=False, dec_mode=DEC_EMO_HEAD_MODE)
            with self.assertRaises(RuntimeError):
                run_loader(file, fn, separate=True, dec_mode="shared")

    def test_07_main_sh_v10_trial1_profile_routing_and_flags(self):
        """检查 main.sh 中 v10_trial1_sep_dec_head 命名分支的固定参数与 results/v10 路由。"""
        script_text = (ROOT / "main.sh").read_text(encoding="utf-8")
        self.assertIn("v10_trial1_sep_dec_head)", script_text)
        self.assertIn("--max_grad_norm 6.0 --arc_margin_ramp_steps $V10_ARC_RAMP --use_separate_dec_emo_head", script_text)
        self.assertIn('v10_*) RESULTS_DIR="results/v10" ;;', script_text)
        self.assertIn("dec_emo_diagnostics.jsonl", script_text)
        self.assertIn("env_metadata.json", script_text)

    def run_main_sh_function(self, marker, name, **values):
        source = (ROOT / "main.sh").read_text(encoding="utf-8")
        block = source.split("# BEGIN_" + marker + "\n", 1)[1].split("# END_" + marker, 1)[0]
        values["pythonpath"] = sys.executable.replace("\\", "/")
        setup = "set -e\nset -o pipefail\n" + "\n".join(k + "=" + shlex.quote(str(v).replace("\\", "/")) for k, v in values.items())
        bash = "D:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"
        return subprocess.run([bash, "-c", setup + "\n" + block + "\n" + name], cwd=str(ROOT),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES=""))

    def test_08_runtime_preflight_records_actual_default_before_training(self):
        """执行预检的真实Python主体；阶段三另在独立进程验证Bash调用及CUDA，避免重复加载CUDA DLL。"""
        source = (ROOT / "main.sh").read_text(encoding="utf-8")
        block = source.split("# BEGIN_RUNTIME_METADATA\n", 1)[1].split("# END_RUNTIME_METADATA", 1)[0]
        python_source = block.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as tmp:
            for mode in ("4G", "24G"):
                out = Path(tmp) / mode
                error = None
                log = io.StringIO()
                with patch.object(torch.cuda, "is_available", return_value=False), \
                     patch.object(sys, "argv", ["-", str(out / "env_metadata.json"), mode, "v10_trial1_sep_dec_head"]), \
                     contextlib.redirect_stdout(log):
                    try:
                        exec(compile(python_source, "main.sh:runtime_preflight", "exec"), {})
                    except RuntimeError as exc:
                        error = exc
                self.assertTrue((out / "env_metadata.json").exists(), str(error))
                info = json.loads((out / "env_metadata.json").read_text(encoding="utf-8"))
                self.assertTrue(info["recorded_before_training"])
                self.assertEqual(info["torch"], torch.__version__)
                self.assertEqual(info["optimizer_zero_grad"][0]["device"], "cpu")
                all_none = all(p["grad_is_none"] for p in info["optimizer_zero_grad"])
                expected_pass = mode == "4G" or all_none
                self.assertEqual(info["runtime_contract_passed"], expected_pass)
                self.assertEqual(error is None, expected_pass, str(error))
                self.assertIn("[Runtime Preflight]", log.getvalue())

    def test_09_runtime_contract_accepts_none_and_rejects_zero_tensor(self):
        """无论测试机 torch 默认值如何，覆盖正式契约的两种行为。"""
        source = (ROOT / "main.sh").read_text(encoding="utf-8")
        block = source.split("# BEGIN_RUNTIME_METADATA\n", 1)[1].split("# END_RUNTIME_METADATA", 1)[0]
        python_source = block.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        fn = next(n for n in ast.parse(python_source).body if isinstance(n, ast.FunctionDef) and n.name == "validate_zero_grad")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.sh:validate_zero_grad", "exec"), namespace)
        validate = namespace["validate_zero_grad"]
        self.assertTrue(validate({"optimizer_zero_grad": [{"grad_is_none": True}]}, "24G", "v10_trial1_sep_dec_head"))
        mixed = {"optimizer_zero_grad": [{"grad_is_none": True}, {"grad_is_none": False}]}
        self.assertFalse(validate(mixed, "24G", "v10_trial1_sep_dec_head"))
        self.assertTrue(validate(mixed, "4G", "v10_trial1_sep_dec_head"))
        self.assertTrue(validate(mixed, "24G", "v9_trial11_B_gnorm6"))

    def test_10_packaging_rejects_missing_v10_required_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as tmp:
            out = Path(tmp) / "output"
            out.mkdir()
            values = dict(OUTPUT_DIR=out, DOWNLOAD_DIR=Path(tmp) / "download", LOG_FILE=Path(tmp) / "absent.log",
                          RESULTS_DIR=Path(tmp), EXP_NAME="v10_trial1_sep_dec_head")
            res = self.run_main_sh_function("ARTIFACT_PACKAGING", "package_artifacts", **values)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("Missing required artifact", res.stderr)
            for f in ("env_metadata.json", "dec_emo_diagnostics.jsonl"):
                (out / f).write_text("{}\n", encoding="utf-8")
            res = self.run_main_sh_function("ARTIFACT_PACKAGING", "package_artifacts", **values)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("Both composite_best and ppl_best", res.stderr)

    def test_11_packaging_hashes_and_copies_without_prefix_initializer(self):
        """小文件夹具验证打包逻辑；真实权重打包另由阶段三烟测核验。"""
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as tmp:
            out, download = Path(tmp) / "output", Path(tmp) / "download"
            out.mkdir()
            for f in ("env_metadata.json", "dec_emo_diagnostics.jsonl", "CASE_1_2.0000_composite_best", "CASE_1_2.0000_ppl_best"):
                (out / f).write_text("fixture: " + f + "\n", encoding="utf-8")
            res = self.run_main_sh_function("ARTIFACT_PACKAGING", "package_artifacts", OUTPUT_DIR=out,
                                           DOWNLOAD_DIR=download, LOG_FILE=Path(tmp) / "absent.log",
                                           RESULTS_DIR=Path(tmp), EXP_NAME="v10_trial1_sep_dec_head")
            self.assertEqual(res.returncode, 0, res.stderr)
            for f in ("env_metadata.json", "dec_emo_diagnostics.jsonl"):
                self.assertEqual((download / f).read_bytes(), (out / f).read_bytes())
            for f, count in (("source_sha256.txt", 9), ("checkpoint_sha256.txt", 2)):
                lines = (download / f).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), count)
                for line in lines:
                    digest, path = line.split(None, 1)
                    self.assertEqual(hashlib.sha256(Path(path.lstrip(" *")).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
