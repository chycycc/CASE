import re
import os

def parse_full_trajectory(log_path):
    try:
        with open(log_path, 'r', encoding='utf-16', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    val_records = []
    test_record = None
    for i, l in enumerate(lines):
        line_clean = l.strip()
        if line_clean.startswith('valid'):
            parts = re.split(r'\s+', line_clean)
            if len(parts) >= 10:
                try:
                    val_records.append({
                        'bow': float(parts[1]),
                        'kl': float(parts[2]),
                        'mim': float(parts[3]),
                        'ctx': float(parts[4]),
                        'ppl': float(parts[5]),
                        'emo_loss': float(parts[8]),
                        'emo_acc': float(parts[9]) * 100
                    })
                except Exception:
                    pass
        elif line_clean.startswith('test'):
            parts = re.split(r'\s+', line_clean)
            if len(parts) >= 10:
                try:
                    test_record = {
                        'bow': float(parts[1]),
                        'kl': float(parts[2]),
                        'mim': float(parts[3]),
                        'ctx': float(parts[4]),
                        'ppl': float(parts[5]),
                        'emo_loss': float(parts[8]),
                        'emo_acc': float(parts[9]) * 100
                    }
                except Exception:
                    pass
    min_val_ppl = min(r['ppl'] for r in val_records) if val_records else None
    max_val_acc = max(r['emo_acc'] for r in val_records) if val_records else None
    min_val_emo_l = min(r['emo_loss'] for r in val_records) if val_records else None
    last_val_ppl = val_records[-1]['ppl'] if val_records else None
    last_val_acc = val_records[-1]['emo_acc'] if val_records else None
    return {
        'num_evals': len(val_records),
        'min_val_ppl': min_val_ppl,
        'max_val_acc': max_val_acc,
        'min_val_emo_l': min_val_emo_l,
        'last_val_ppl': last_val_ppl,
        'last_val_acc': last_val_acc,
        'test': test_record,
        'records': val_records
    }

if __name__ == '__main__':
    logs = [
        ('V5 Trial 1 (div=2.0x)', 'train_v5_trial1.log'),
        ('V5 Trial 3a (div=1.8x)', 'train_v5_trial3a.log'),
        ('V5 Trial 3b (div=2.2x)', 'train_v5_trial3b.log'),
        ('V5 Trial 3c (div=2.5x)', 'train_v5_trial3c.log')
    ]
    for name, lp in logs:
        info = parse_full_trajectory(lp)
        num_ev = info['num_evals']
        min_p = info['min_val_ppl']
        max_a = info['max_val_acc']
        min_el = info['min_val_emo_l']
        last_p = info['last_val_ppl']
        last_a = info['last_val_acc']
        t_p = info['test']['ppl'] if info['test'] else 0
        t_a = info['test']['emo_acc'] if info['test'] else 0
        t_el = info['test']['emo_loss'] if info['test'] else 0
        print(f"=== {name} ===")
        print(f"  Valid 评估次数: {num_ev}")
        print(f"  Min Val PPL: {min_p:.2f} | Max Val Emo Acc: {max_a:.2f}% | Min Val EMO Loss: {min_el:.4f}")
        print(f"  Last Val PPL: {last_p:.2f} | Last Val Emo Acc: {last_a:.2f}%")
        print(f"  Test PPL: {t_p:.2f} | Test Emo Acc: {t_a:.2f}% | Test EMO Loss: {t_el:.4f}")
        print()
