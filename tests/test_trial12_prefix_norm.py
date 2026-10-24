"""验证前缀标准化、零门控原始梯度对照及观测状态隔离。"""
import contextlib
import copy
import hashlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
with contextlib.redirect_stdout(io.StringIO()):
    from src.models.common import DecoderLayer, LayerNorm
from src.utils.prefix_runtime import (PREFIX_NORM_MODE, FROZEN_INIT_SHA256, checkpoint_prefix_norm,
    capture_rng_state, assert_rng_equal, isolated_rng, isolated_model_observation, emotion_selection_summary)


class PrefixNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_formula_and_default_preserve_legacy_exactly(self):
        layer = LayerNorm(12)
        with torch.no_grad():
            layer.gamma.copy_(torch.randn(12)); layer.beta.copy_(torch.randn(12))
        x = torch.randn(4, 12)
        expected = layer.gamma * (x-x.mean(-1, keepdim=True))/(x.std(-1, keepdim=True)+1e-6)
        self.assertTrue(torch.equal(layer(x), expected+layer.beta))
        self.assertTrue(torch.equal(layer(x, add_bias=False), expected))
        layer(x, add_bias=False).square().sum().backward()
        self.assertIsNone(layer.beta.grad)
        self.assertGreater(layer.gamma.grad.abs().sum().item(), 0)

    def test_zero_gate_outputs_and_raw_branch_gradients_equal(self):
        """比较PCGrad与裁剪之前的梯度；这两项全局运算可能传播门控差异。"""
        for initializer in (torch.randn(4, 12), torch.eye(4, 12)):
            kw = dict(use_proto_prefix=True, use_proto_prefix_gate=True, proto_init=initializer)
            off = DecoderLayer(12, 12, 12, 12, 2, None, **kw)
            state = torch.get_rng_state().clone()
            on = copy.deepcopy(off)
            on.use_proto_prefix_norm = True
            on.register_buffer("prefix_norm_version", torch.tensor(1))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            x = torch.randn(2, 5, 12); memory = torch.randn(2, 6, 12)
            mask = torch.zeros(2, 1, 6, dtype=torch.bool)
            causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1).expand(2, -1, -1)
            inputs = (x, memory, None, (mask, causal), memory, mask, memory, mask)
            for loss_fn in (lambda y: y.square().mean(), lambda y: y.sin().sum()):
                off.zero_grad(set_to_none=True); on.zero_grad(set_to_none=True)
                a = off(inputs)[0]; b = on(inputs)[0]
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(loss_fn(a), loss_fn(b)))
                loss_fn(a).backward(); loss_fn(b).backward()
                for (name, pa), (_, pb) in zip(off.named_parameters(), on.named_parameters()):
                    if name == "prefix_gate":
                        self.assertFalse(torch.equal(pa.grad, pb.grad))
                    elif pa.grad is None:
                        self.assertIsNone(pb.grad)
                    else:
                        self.assertTrue(torch.equal(pa.grad, pb.grad), name)
                self.assertEqual(on.prefix_embed.grad.abs().max().item(), 0)
            with torch.no_grad():
                on.prefix_gate.fill_(.01)
            on.zero_grad(set_to_none=True); on(inputs)[0].square().mean().backward()
            self.assertGreater(on.prefix_embed.grad.abs().sum().item(), 0)

    def test_norm_requires_gate_and_marker_values_are_checked(self):
        with self.assertRaises(ValueError):
            DecoderLayer(12, 12, 12, 12, 2, None, use_proto_prefix=True, use_proto_prefix_norm=True)
        state = {'prefix_norm_mode': PREFIX_NORM_MODE, 'prefix_attention_mode': 'separate_softmax_residual',
                 'model': {'decoder.dec.0.prefix_embed': torch.ones(4, 12),
                           'decoder.dec.0.prefix_gate': torch.zeros(2),
                           'decoder.dec.0.prefix_norm_version': torch.tensor(1)}}
        self.assertTrue(checkpoint_prefix_norm(state))
        for bad in (torch.tensor(0), torch.tensor(2), torch.tensor(1.0), torch.tensor([1])):
            state['model']['decoder.dec.0.prefix_norm_version'] = bad
            with self.assertRaises(RuntimeError): checkpoint_prefix_norm(state)

    def test_frozen_both_groups_replay_exact_tensor_without_rng(self):
        from src.models.CASE.model import CASE
        from src.utils.config import config
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch') as tmp:
            for mode in ('proto', 'random'):
                init = torch.randn(4, 300).numpy()
                path = Path(tmp)/('input_'+mode+'.npz'); np.savez(path, init=init)
                expected = hashlib.sha256(init.tobytes()).hexdigest()
                with patch.dict(FROZEN_INIT_SHA256, {mode: expected}), patch.multiple(config,
                    use_proto_prefix=True, test=False, proto_prefix_len=4, hidden_dim=300,
                    proto_prefix_source='frozen', proto_prefix_init=mode,
                    proto_prefix_init_npz=str(path), save_path=str(Path(tmp)/mode)):
                    holder = CASE.__new__(CASE); nn.Module.__init__(holder)
                    holder.use_proto_prefix_gate = holder.use_proto_prefix_norm = True
                    before = capture_rng_state()
                    with contextlib.redirect_stdout(io.StringIO()): result = holder._build_proto_prefix_init()
                    assert_rng_equal(before, capture_rng_state())
                    self.assertTrue(np.array_equal(result.numpy(), init))


class ObservationIsolationTests(unittest.TestCase):
    def test_all_rng_sources_and_next_training_sampler_restore_even_on_error(self):
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
        from torch.utils.data import DataLoader
        before = capture_rng_state()
        expected = list(DataLoader(list(range(20)), shuffle=True, batch_size=4))
        random.setstate(before[0]); np.random.set_state(before[1]); torch.set_rng_state(before[2])
        if before[3]: torch.cuda.set_rng_state_all(before[3])
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with isolated_rng():
                random.random(); np.random.rand(); torch.rand(2)
                list(DataLoader(list(range(20)), shuffle=False, batch_size=4))
                if before[3]: torch.rand(2, device='cuda')
                raise ValueError('sentinel')
        assert_rng_equal(before, capture_rng_state())
        actual = list(DataLoader(list(range(20)), shuffle=True, batch_size=4))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_legacy_eval_mutations_and_gradients_are_restored(self):
        model = nn.Module(); model.p = nn.Parameter(torch.ones(3))
        model.criterion = nn.NLLLoss(weight=torch.ones(3)); model.criterion.eval()
        model.word_freq = np.array([1., 2., 3.]); model.p.grad = torch.ones_like(model.p)
        original_weight = model.criterion.weight; original_grad = model.p.grad
        before = capture_rng_state()
        with isolated_model_observation(model):
            model.word_freq += 10
            model.criterion.weight = torch.arange(3.)
            model.p.grad.zero_(); model.zero_grad(set_to_none=True)
            model.current_pred_emotion = np.array([5])
            torch.rand(3); np.random.rand(); random.random()
        assert_rng_equal(before, capture_rng_state())
        self.assertTrue(model.training); self.assertFalse(model.criterion.training)
        self.assertIs(model.p.grad, original_grad); self.assertTrue(torch.equal(model.p.grad, torch.ones(3)))
        self.assertIs(model.criterion.weight, original_weight)
        self.assertTrue(np.array_equal(model.word_freq, [1., 2., 3.]))
        self.assertFalse(hasattr(model, 'current_pred_emotion'))

    def test_label_gap_uses_example_weights_and_fixed_count_filter(self):
        # 类别样本数5与15：总体分布按实际频数加权，不能按类别等权混合。
        attention = torch.tensor([[[1., 0.]]]*5 + [[[0., 1.]]]*15 + [[[.5, .5]]])
        result = emotion_selection_summary(attention, [0]*5+[1]*15+[2])
        expected = -(.25*np.log(.25)+.75*np.log(.75))
        self.assertAlmostEqual(result['entropy_gap_by_head'][0], expected)
        self.assertEqual(result['covered_n'], 20); self.assertEqual(result['total_n'], 21)
        self.assertEqual(len(result['classes']), 3)


if __name__ == '__main__': unittest.main()
