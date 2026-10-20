# -*- coding: utf-8 -*-
"""Dependency-light tests of actual evaluation scope/counting functions."""
import ast
import contextlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
torch = SimpleNamespace(no_grad=contextlib.nullcontext)

def source_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]

class EvaluationScopeTests(unittest.TestCase):
    def test_test_scope_counts_short_last_batch_and_names_correct_ppl_formula(self):
        class Loader(list):
            batch_size = 2
        loader = Loader([{"program_label": SimpleNamespace(size=lambda axis: 2)},
                         {"program_label": SimpleNamespace(size=lambda axis: 1)}])
        losses = iter([math.log(2), math.log(8)])
        def forward(*args, **kwargs):
            ctx = next(losses)
            return (0, 0, 0, ctx, math.exp(ctx), 0, 0, 0, 0, 0, 0)
        namespace = dict(np=np, math=math, torch=torch, tqdm=lambda data, **kw: data)
        evaluate = source_function("src/scripts/eval_pipeline.py", "evaluate_test_set", namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(SimpleNamespace(train_one_batch=forward), loader)
        self.assertEqual(result["Test_PPL"], 4.0)
        self.assertEqual(result["_scope"]["n_samples"], 3)
        self.assertEqual(result["_scope"]["aggregation"], "macro: exp(mean per-batch token CE)")

    def test_generation_reuses_single_sample_loader_and_records_actual_count(self):
        class Loader(list):
            batch_size = 1
        data = Loader([{}, {}])
        decode = lambda *a, **kw: ["a b"]
        model = SimpleNamespace(decoder_greedy=decode, decoder_sampling=decode)
        namespace = dict(torch=torch, np=np, tqdm=lambda data, **kw: data,
                         Translator=lambda *a: SimpleNamespace(beam_search=decode),
                         word_tokenize=str.split, calc_distinct_n=lambda *a: 0,
                         prepare_data_seq=lambda **kw: self.fail("must reuse batch=1 loader"))
        evaluate = source_function("src/scripts/eval_pipeline.py", "evaluate_generation_diversity", namespace)
        for requested, expected in ((1, 1), (100, 2), (0, 2)):
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(model, None, data, requested)
            self.assertEqual(result["_scope"]["n_samples"], expected)



if __name__ == "__main__":
    unittest.main()
