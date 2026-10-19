# -*- coding: utf-8 -*-
"""离线复核 Emo_Hit 指标：用 results.txt 中已有的 5255 条生成结果，无需 GPU"""
import re, json, collections
from pathlib import Path

LEX = {
 "surprised": {"surprised","surprise","surprising","shocked","shocking","unexpected","astonished","unbelievable","wonder","whoa"},
 "excited": {"excited","exciting","excitement","thrilled","stoked","awesome","great","wonderful","yay","pumped"},
 "annoyed": {"annoyed","annoying","irritated","bothered","frustrated","bugged","aggravated","hate","ugh"},
 "proud": {"proud","pride","accomplishment","achievement","proudly","congrats","congratulations","honor"},
 "angry": {"angry","mad","pissed","furious","outraged","anger","rage","livid"},
 "sad": {"sad","depressed","unhappy","sorrow","crying","tears","heartbroken","down","grief"},
 "grateful": {"grateful","thank","thanks","thankful","appreciate","appreciated","gratitude","blessed"},
 "lonely": {"lonely","alone","isolated","isolation","lonesome","nobody","friendless"},
 "impressed": {"impressed","impressive","awesome","amazing","admire","incredible","cool"},
 "afraid": {"afraid","scared","fear","fearful","frightened","creepy","spooky"},
 "disgusted": {"disgusted","disgusting","gross","sick","nasty","revolting","yuck","eww"},
 "confident": {"confident","confidence","sure","certain","positive","trust","strong"},
 "terrified": {"terrified","terrifying","terror","horrified","petrified","panic","nightmare"},
 "hopeful": {"hopeful","hope","hoping","optimistic","praying","positive","wish"},
 "anxious": {"anxious","anxiety","nervous","worry","worried","worrying","stress","stressed"},
 "disappointed": {"disappointed","disappointing","disappointment","letdown","bummer","shame","sadly"},
 "joyful": {"joyful","joy","happy","happiness","glad","delighted","cheerful","smile","smiling"},
 "prepared": {"prepared","ready","preparation","organized","set","plan"},
 "guilty": {"guilty","guilt","fault","regret","blame","bad","apologize","sorry"},
 "furious": {"furious","fury","infuriated","enraged","mad","angry","screaming"},
 "nostalgic": {"nostalgic","nostalgia","remember","memories","memory","reminisce","back then","old times","past"},
 "jealous": {"jealous","jealousy","envy","envious","covet"},
 "anticipating": {"anticipating","anticipation","expecting","wait","waiting","forward","countdown"},
 "embarrassed": {"embarrassed","embarrassing","embarrassment","humiliated","awkward","cringe","blush","foolish"},
 "content": {"content","satisfied","contentment","peaceful","peace","calm","relax","chilled"},
 "devastated": {"devastated","devastating","crushed","destroyed","heartbroken","ruined","wrecked"},
 "sentimental": {"sentimental","sentiment","miss","memory","memories","remember","cherish","fond","special"},
 "caring": {"caring","care","concerned","helpful","sweet","kind","support","hug","love"},
 "trusting": {"trusting","trust","believe","faith","reliable","honest","depend"},
 "ashamed": {"ashamed","shame","disgrace","embarrassed","humiliated","unworthy"},
 "apprehensive": {"apprehensive","uneasy","nervous","hesitant","concerned","dread","unsure","fear"},
 "faithful": {"faithful","loyal","loyalty","devoted","faith","true","committed"},
}

def parse_results(path):
    rows = []
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\nEmotion:\s*", "\n" + txt)
    for b in blocks[1:]:
        lines = b.split("\n")
        emo = lines[0].strip()
        rec = {"emotion": emo, "beam": "", "greedy": "", "ref": ""}
        for ln in lines[1:8]:
            if ln.startswith("Beam:"): rec["beam"] = ln[5:].strip()
            elif ln.startswith("Greedy:"): rec["greedy"] = ln[7:].strip()
            elif ln.startswith("Ref:"): rec["ref"] = ln[4:].strip()
        if rec["beam"] or rec["greedy"]:
            rows.append(rec)
    return rows

def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))

def emo_hit(rows, key, n=None):
    rs = rows[:n] if n else rows
    hit, total, hit_words = 0, 0, collections.Counter()
    for r in rs:
        lex = LEX.get(r["emotion"].lower())
        if not lex: continue
        t = toks(r[key]); total += 1
        inter = t & lex
        if inter:
            hit += 1
            for w in inter: hit_words[w] += 1
    return (hit/total*100 if total else 0.0), total, hit_words

base = Path(r"E:\github\CASE\results\cloud_archive")
runs = {
 "T10":  base/"v9_trial10/v9_trial10_results.txt",
 "T11":  base/"v9_trial11/v9_trial11_seed13_results.txt",
 "T11-B": base/"v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_results.txt",
 "T11-A": base/"v9_trial11_A_nopcam/v9_trial11_A_nopcam_results.txt",
}
print("=== 解析结果 ===")
data = {}
for name, p in runs.items():
    if not p.exists():
        print(f"  {name}: 文件缺失 {p}"); continue
    rows = parse_results(p)
    data[name] = rows
    print(f"  {name}: 解析出 {len(rows)} 条（含 Emotion 标签+生成）")

print("\n=== Emo_Hit（全部已解析样本）===")
print(f"{'配置':<7} {'样本':>6} {'Greedy':>9} {'Beam':>9}")
for name, rows in data.items():
    for key in ("greedy","beam"):
        pass
    g, n, _ = emo_hit(rows, "greedy")
    b, n2, _ = emo_hit(rows, "beam")
    print(f"{name:<7} {n:>6} {g:>8.2f}% {b:>8.2f}%")

print("\n=== 前 500 条（与 eval_pipeline 的 gen_samples=500 同口径）===")
print(f"{'配置':<7} {'Greedy':>9} {'Beam':>9}")
for name, rows in data.items():
    g, _, _ = emo_hit(rows, "greedy", 500)
    b, _, _ = emo_hit(rows, "beam", 500)
    print(f"{name:<7} {g:>8.2f}% {b:>8.2f}%")

print("\n=== T11-B 实际命中的词（Greedy，前 500）—— 检验是否有「假命中」===")
_, n, hw = emo_hit(data["T11-B"], "greedy", 500)
for w, c in hw.most_common(15):
    print(f"   {w:<14} {c}")

print("\n=== T11-B 目标情绪分布（前 500）===")
cnt = collections.Counter(r["emotion"] for r in data["T11-B"][:500])
print("  " + ", ".join(f"{k}:{v}" for k,v in cnt.most_common(10)))
