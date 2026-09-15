# -*- coding: utf-8 -*-
import re

try:
    with open("train_v8_trial2.log", "r", encoding="utf-16le", errors="ignore") as f:
        text = f.read()
    if len(text) < 100:
        with open("train_v8_trial2.log", "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
except Exception:
    with open("train_v8_trial2.log", "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

pattern = re.findall(r"valid\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", text)
print(f"Total validation steps found: {len(pattern)}")
for i, m in enumerate(pattern):
    step = (i + 1) * 2000
    bow, kl, mim, ctx, ppl, str_l, str_a, emo_l, emo_a = m
    print(f"Step {step:5d}: PPL={float(ppl):.4f} | EMO_acc={float(emo_a)*100:.2f}% | EMO_loss={float(emo_l):.4f} | BOW={float(bow):.4f} | KL={float(kl):.4f} | MIM={float(mim):.4f}")
