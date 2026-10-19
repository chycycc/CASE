import json

t10 = json.load(open('results/cloud_archive/v9_trial10/eval_metrics.json'))
t11 = json.load(open('results/cloud_archive/v9_trial11/eval_metrics.json'))
t11b = json.load(open('results/cloud_archive/v9_trial11_B_gnorm6/eval_metrics.json'))

print('=== Test Metrics ===')
for k in t10['test_metrics']:
    v10 = t10['test_metrics'].get(k)
    v11 = t11['test_metrics'].get(k)
    v11b = t11b['test_metrics'].get(k)
    diff = round(v11b - v11, 4) if (v11b is not None and v11 is not None) else 'N/A'
    print(f'{k:20s}: T10={v10:<8} | T11={v11:<8} | T11-B={v11b:<8} | Delta(T11B - T11)={diff}')

print('\n=== Manifold Metrics ===')
for k in t10['manifold_metrics']:
    if k == 'Plot_Path': 
        continue
    v10 = t10['manifold_metrics'].get(k)
    v11 = t11['manifold_metrics'].get(k)
    v11b = t11b['manifold_metrics'].get(k)
    diff = round(v11b - v11, 4) if (v11b is not None and v11 is not None) else 'N/A'
    print(f'{k:20s}: T10={v10:<8} | T11={v11:<8} | T11-B={v11b:<8} | Delta(T11B - T11)={diff}')

print('\n=== Generation Metrics ===')
for mode in ['Greedy', 'Beam', 'Sampling', 'Sampling_0.8']:
    print(f'-- {mode} --')
    for k in t10['generation_metrics'][mode]:
        v10 = t10['generation_metrics'][mode].get(k)
        v11 = t11['generation_metrics'][mode].get(k)
        v11b = t11b['generation_metrics'][mode].get(k)
        diff = round(v11b - v11, 4) if (v11b is not None and v11 is not None) else 'N/A'
        print(f'{k:18s}: T10={v10:<8} | T11={v11:<8} | T11-B={v11b:<8} | Delta(T11B - T11)={diff}')
