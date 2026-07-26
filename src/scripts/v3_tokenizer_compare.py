"""
v3 P0 验证脚本：对比 split() vs word_tokenize 对 Distinct 的影响
直接读取 save/epcl_v2_step3_pre_gate/results.txt 中的 Greedy 生成结果
"""
from nltk import word_tokenize

def calc_distinct_n_with_tokenizer(n, candidates, tokenizer_fn):
    """通用 Distinct-n 计算，接受不同的分词函数"""
    ngram_set = {}
    total = 0
    tokenized = [tokenizer_fn(c) for c in candidates]
    for sentence in tokenized:
        for i in range(len(sentence) - n + 1):
            key = tuple(sentence[i : i + n])
            ngram_set[key] = 1
            total += 1
    return len(ngram_set) / (total + 1e-16)

def extract_greedy_from_results(filepath):
    """从 results.txt 中提取所有 Greedy 生成的句子"""
    cands = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Greedy:"):
                text = line[len("Greedy:"):].strip()
                if text:
                    cands.append(text)
    return cands

if __name__ == "__main__":
    filepath = "save/epcl_v2_step3_pre_gate/results.txt"
    cands = extract_greedy_from_results(filepath)
    print(f"共提取 {len(cands)} 条 Greedy 生成结果\n")

    # 方法 A: split() — 当前使用的方式
    d1_split = calc_distinct_n_with_tokenizer(1, cands, lambda s: s.split())
    d2_split = calc_distinct_n_with_tokenizer(2, cands, lambda s: s.split())

    # 方法 B: word_tokenize — 原论文使用的方式
    d1_wt = calc_distinct_n_with_tokenizer(1, cands, word_tokenize)
    d2_wt = calc_distinct_n_with_tokenizer(2, cands, word_tokenize)

    print("=" * 60)
    print(f"{'指标':<12} {'split()':<15} {'word_tokenize':<15} {'差值':<10}")
    print("=" * 60)
    print(f"{'Dist-1':<12} {d1_split*100:<15.4f} {d1_wt*100:<15.4f} {(d1_wt-d1_split)*100:<+10.4f}")
    print(f"{'Dist-2':<12} {d2_split*100:<15.4f} {d2_wt*100:<15.4f} {(d2_wt-d2_split)*100:<+10.4f}")
    print("=" * 60)
    print(f"\n原论文 Dist-2 = 4.01%")
    print(f"word_tokenize 口径下我们的 Dist-2 = {d2_wt*100:.4f}%")
