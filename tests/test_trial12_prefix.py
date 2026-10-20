# -*- coding: utf-8 -*-
"""Trial 12 regression checks. Run: python tests/test_trial12_prefix.py -v.

AST loading executes the actual entry-point functions without importing their
CLI side effects or loading a full dataset; attention checks use real modules.
"""
import argparse
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
with contextlib.redirect_stdout(io.StringIO()):
    from src.models.common import MultiHeadAttention, DecoderLayer


def source_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class PrefixAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.attn = MultiHeadAttention(12, 12, 12, 12, 2, None, 0.0).eval()
        self.x = torch.randn(2, 5, 12)
        self.prefix = torch.randn(4, 12, requires_grad=True)
        self.mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).expand(2, -1, -1)

    def test_prefix_matches_explicit_kv_concatenation(self):
        actual, logits = self.attn(self.x, self.x, self.x, self.mask, extra_kv=self.prefix)
        kv = torch.cat((self.prefix.unsqueeze(0).expand(2, -1, -1), self.x), dim=1)
        mask = torch.cat((torch.zeros(2, 5, 4, dtype=torch.bool), self.mask), dim=-1)
        expected, expected_logits = self.attn(self.x, kv, kv, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(tuple(logits.shape), (2, 5, 9))
        probabilities = []
        handle = self.attn.dropout.register_forward_pre_hook(lambda module, inputs: probabilities.append(inputs[0]))
        self.attn(self.x, self.x, self.x, self.mask, extra_kv=self.prefix)
        handle.remove()
        weights = probabilities[0]
        self.assertTrue((weights[..., :4].sum(dim=-1) > 0).all().item())
        self.assertTrue((weights[..., 4:].masked_select(self.mask.unsqueeze(1)) == 0).all().item())

    def test_prefix_keeps_future_tokens_masked(self):
        first, _ = self.attn(self.x, self.x, self.x, self.mask, extra_kv=self.prefix)
        changed = self.x.clone()
        changed[:, 3:] += 100
        second, _ = self.attn(changed, changed, changed, self.mask, extra_kv=self.prefix)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=0)

    def test_decoder_prefix_has_nonzero_gradient(self):
        layer = DecoderLayer(12, 12, 12, 12, 2, None, 0.0, 0.0, 0.0,
                             use_proto_prefix=True, proto_prefix_len=4,
                             proto_init=self.prefix.detach()).eval()
        memory = torch.randn(2, 6, 12)
        memory_mask = torch.zeros(2, 1, 6, dtype=torch.bool)
        inputs = (self.x, memory, None, (memory_mask, self.mask),
                  memory, memory_mask, memory, memory_mask)
        out, *_ = layer(inputs)
        out.square().mean().backward()
        grad = layer.prefix_embed.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.norm().item(), 0)
        before = out.detach()
        with torch.no_grad():
            layer.prefix_embed.add_(0.5)
            after, *_ = layer(inputs)
        self.assertGreater((before - after).abs().max().item(), 0)
        disabled = DecoderLayer(12, 12, 12, 12, 2, None, 0.0, 0.0, 0.0)
        self.assertFalse(any("prefix_embed" in k for k in disabled.state_dict()))

    def test_actual_make_model_preserves_prior_and_random_initial_values(self):
        class SmallCASE(nn.Module):
            def __init__(self, init):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.prefix_embed = nn.Parameter(init.clone())
                self.other = nn.Linear(12, 12, bias=False)
        cfg = SimpleNamespace(test=False, model="case", device="cpu", pretrain_emb=True)
        prior = self.prefix.detach()
        random = torch.randn_like(prior)
        results = []
        for init in (prior, random):
            namespace = dict(config=cfg, torch=torch, CASE=lambda *a, **kw: SmallCASE(init),
                             xavier_uniform_=nn.init.xavier_uniform_,
                             count_parameters=lambda m: sum(p.numel() for p in m.parameters()))
            make_model = source_function("main.py", "make_model", namespace)
            torch.manual_seed(13)
            with contextlib.redirect_stdout(io.StringIO()):
                model = make_model(None, 32, 8)
            self.assertTrue(torch.equal(model.decoder.prefix_embed, init))
            results.append(model)
        self.assertFalse(torch.equal(results[0].decoder.prefix_embed, results[1].decoder.prefix_embed))
        self.assertTrue(torch.equal(results[0].other.weight, results[1].other.weight))


    def test_random_initialization_uses_local_rng_and_matches_row_norms(self):
        from src.models.CASE.model import CASE
        from src.utils.config import config
        e = torch.nn.functional.normalize(torch.randn(4, 300), dim=1)
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
            path = str(Path(directory) / "init.npz")
            np.savez(path, init=e.numpy())
            with patch.multiple(config, use_proto_prefix=True, test=False,
                                proto_prefix_len=4, hidden_dim=300,
                                proto_prefix_init="random", proto_prefix_init_npz=path):
                before = torch.get_rng_state().clone()
                with contextlib.redirect_stdout(io.StringIO()):
                    random = CASE._build_proto_prefix_init(None)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertFalse(torch.equal(e, random))
                self.assertLess((e.norm(dim=1) - random.norm(dim=1)).abs().max().item(), 1e-5)

    def test_case_prints_explicit_false_without_t12_tag(self):
        from src.models.CASE.model import CASE
        from src.utils.config import config
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
            with patch.multiple(config, use_proto_prefix=False, test=False, device="cpu",
                                pretrain_emb=False, save_path=directory, dataset="ED"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    model = CASE(SimpleNamespace(n_words=64), emotion_num=32, strategy_num=8)
                self.assertIn("[Prefix Status] use_proto_prefix=False", output.getvalue())
                self.assertNotIn("[V9 T12]", output.getvalue())
                self.assertFalse(any("prefix_embed" in key for key in model.state_dict()))


class EvaluationAuditTests(unittest.TestCase):
    def loader(self, file, fn, prefix, unexpected=()):
        cfg = SimpleNamespace(device="cpu", use_proto_prefix=not prefix)
        state = {"model": {"epcl_criterion.prototypes": torch.zeros(32, 300),
                           "emotion_linear.weight": torch.zeros(32, 300)}}
        if prefix:
            state["model"]["decoder.dec.0.prefix_embed"] = torch.zeros(4, 300)
        class Model:
            epcl_criterion = object()
            def to(self, device): return self
            def eval(self): pass
            def load_state_dict(self, sd, strict):
                return [], list(unexpected)
        namespace = dict(os=os, config=cfg, torch=torch, CASE=lambda *a, **kw: Model(),
                         PROJECT_ROOT=str(ROOT), set_seed=lambda: None,
                         prepare_data_seq=lambda **kw: (None, None, None, None, 32, 8))
        loader = source_function(file, fn, namespace)
        with tempfile.NamedTemporaryFile(dir=ROOT / "scratch") as f:
            with patch.object(torch, "load", return_value=state), contextlib.redirect_stdout(io.StringIO()):
                result = loader(f.name)
        return cfg, result

    def test_ncm_detects_real_prefix_key_and_resets_false(self):
        for prefix in (True, False):
            cfg, _ = self.loader("src/scripts/eval_prototype_ncm.py", "safe_load_checkpoint", prefix)
            self.assertEqual(cfg.use_proto_prefix, prefix)
            if prefix:
                self.assertEqual(cfg.proto_prefix_len, 4)

    def test_pipeline_rejects_unexpected_critical_weights(self):
        for key in ("decoder.rogue", "pcam.rogue", "epcl_criterion.rogue"):
            with self.assertRaises(RuntimeError):
                self.loader("src/scripts/eval_pipeline.py", "load_v8_model", True, [key])

    def test_skipped_evaluation_does_not_fabricate_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.touch()
            report = Path(directory) / "report.txt"
            test_set = SimpleNamespace(dataset=list(range(5255)), batch_size=1)
            cfg = SimpleNamespace()
            namespace = dict(argparse=argparse, os=os, glob=__import__("glob"), torch=torch,
                             config=cfg, time=time, json=json, np=np, math=math,
                             load_v8_model=lambda *a, **kw: (None, None, test_set))
            main = source_function("src/scripts/eval_pipeline.py", "main", namespace)
            args = ["eval_pipeline.py", "--model_path", str(checkpoint), "--results_path", str(report),
                    "--skip_test_eval", "--skip_manifold", "--skip_generation"]
            with patch.object(sys, "argv", args), patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                main()
            data = json.loads((Path(directory) / "eval_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(data["test_metrics"], {})
            self.assertEqual(data["manifold_metrics"], {})
            self.assertEqual(data["evaluation_protocol"]["test"]["evaluated_n"], 0)
            self.assertEqual(data["evaluation_protocol"]["test"]["dataset_n"], 5255)
            self.assertNotIn("Test MIM_loss", report.read_text(encoding="utf-8"))



if __name__ == "__main__":
    unittest.main()
