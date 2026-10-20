import sys, os, traceback, io

sys.path.insert(0, os.path.abspath('.'))

old_stdout = sys.stdout
sys.stdout = io.StringIO()
from src.utils.config import config
sys.stdout = old_stdout

print("Config imported successfully.")

# Mock parameters
config.dataset = "ED"
config.use_proto_prefix = False
config.hidden_dim = 300
config.emb_dim = 300
config.hop = 1
config.heads = 2
config.depth = 100
config.filter = 300
config.dropout = 0.1
config.PAD_idx = 1
config.lr = 0.0001
config.noam = 0
config.weight_sharing = False
config.use_pcam = True
config.use_emo_bias = True
config.use_mcp = True
config.device = "cpu"
config.test = False

class MockVocab:
    def __init__(self):
        self.word2idx = {"<pad>": 1, "<unk>": 0, "<s>": 2, "</s>": 3}
        for i in range(1000):
            self.word2idx[f"word_{i}"] = i + 4
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word2index = self.word2idx
        self.n_words = len(self.word2idx)
    def __len__(self):
        return len(self.word2idx)

vocab = MockVocab()

from src.models.CASE.model import CASE

print("\n--- Testing Prefix OFF ---")
try:
    model_off = CASE(vocab, emotion_num=32, strategy_num=8)
    off_params = sum(p.numel() for p in model_off.parameters())
    print(f"Prefix OFF total params: {off_params:,}")
except Exception as e:
    traceback.print_exc()

print("\n--- Testing Prefix ON (T10 prior) ---")
config.use_proto_prefix = True
config.proto_prefix_len = 4
config.proto_prefix_init = "proto"
config.proto_prefix_source = "ckpt"
config.save_path = "save/test_trial12"
config.proto_prefix_prior_path = "results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best"

try:
    model_on = CASE(vocab, emotion_num=32, strategy_num=8)
    on_params = sum(p.numel() for p in model_on.parameters())
    print(f"Prefix ON total params: {on_params:,}")
    print(f"Param delta: {on_params - off_params} (expected: 4 * 300 = 1,200)")
except Exception as e:
    traceback.print_exc()

print("\n--- Testing Prefix ON Random (T12-C mock) ---")
config.proto_prefix_init = "random"
config.proto_prefix_init_npz = "save/test_trial12/proto_prefix_init.npz"

try:
    model_rand = CASE(vocab, emotion_num=32, strategy_num=8)
    rand_params = sum(p.numel() for p in model_rand.parameters())
    print(f"Prefix Random total params: {rand_params:,}")
    print(f"Param delta: {rand_params - off_params} (expected: 1,200)")
except Exception as e:
    traceback.print_exc()
