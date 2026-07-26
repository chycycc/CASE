"""V3 实验结果评估脚本：直接读取指定 results.txt 并计算 Dist-1/2"""
import sys
from nltk import word_tokenize

def calc_distinct_n(n, candidates):
    d = {}
    total = 0
    candidates_tok = [word_tokenize(c) for c in candidates]
    for sentence in candidates_tok:
        for i in range(len(sentence) - n + 1):
            key = tuple(sentence[i:i+n])
            d[key] = 1
            total += 1
    score = len(d) / (total + 1e-16)
    return score

if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "save/epcl_v3_dual_anchor/results.txt"
    
    with open(results_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 头部指标
    vals = lines[1].strip().split('\t')
    ppl = float(vals[5])
    emo_acc = float(vals[9])
    
    # Greedy 生成结果
    cands = []
    for line in lines:
        if line.startswith('Greedy:'):
            exp = line[len('Greedy:'):].strip()
            cands.append(exp)
    
    dist1 = calc_distinct_n(1, cands)
    dist2 = calc_distinct_n(2, cands)
    
    print(f"=== V3 Evaluation Results ({results_path}) ===")
    print(f"PPL:      {ppl:.4f}")
    print(f"Emo Acc:  {emo_acc*100:.2f}%")
    print(f"Samples:  {len(cands)}")
    print(f"Dist-1:   {dist1*100:.4f}%")
    print(f"Dist-2:   {dist2*100:.4f}%")
