import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scripts.eval_prototype_ncm import safe_load_checkpoint, extract_features_and_logits
model, vocab, dev, test = safe_load_checkpoint(sys.argv[1], batch_size=32)
assert model.use_proto_prefix and model.decoder.dec[0].prefix_embed.shape == (4, 300)
logits, projected, labels = extract_features_and_logits(model, [next(iter(test))], split_name="smoke_one_sample")
assert logits.shape == (1, 32) and labels.shape == (1,)
assert torch.isfinite(logits).all() and torch.isfinite(projected).all()
result = {"checkpoint": sys.argv[1], "scope": "one-sample loading/forward smoke only", "n": 1, "batch_size": test.batch_size, "prefix_loaded": True, "logits_shape": list(logits.shape)}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
