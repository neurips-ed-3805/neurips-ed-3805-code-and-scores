"""
Meta-evaluation (Spearman / Pearson / Kendall) for all
11 judges on the 5 standard ranges across all datasets/metrics.
"""

import os, re, json, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau

warnings.filterwarnings("ignore")

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_DIR  = os.path.join(_ROOT, "scores")    
REPORTS     = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

STD_RANGES = [(0,1),(1,3),(1,5),(0,10),(0,100)]

CANONICAL = {
    "summeval": {"coh":(1,5),"con":(1,5),"flu":(1,5),"rel":(1,5)},
    "newsroom": {"coh":(1,5),"flu":(1,5),"inf":(1,5),"rel":(1,5)},
    "tc":       {"coh":(0,1),"eng":(1,3),"gro":(0,1),"nat":(1,3)},
    "wp_a":     {"coh":(1,5),"enj":(1,5),"gra":(1,5),"rel":(1,5)},
}

DATASETS = {
    "summeval": {"metrics": {"coh":"coherence","con":"consistency","flu":"fluency","rel":"relevance"},    "ds_type":"summeval"},
    "newsroom": {"metrics": {"coh":"coherence","flu":"fluency","inf":"informativeness","rel":"relevance"},"ds_type":"newsroom"},
    "tc":       {"metrics": {"coh":"Understandable","eng":"Engaging","gro":"Uses Knowledge","nat":"Natural"}, "ds_type":"tc"},
    "wp_a":     {"metrics": {"coh":"cohesive","enj":"enjoy","gra":"grammar","rel":"relevant"},           "ds_type":"wp_a"},
}

ALL_JUDGES = [
    "gemma2_9b","gemma3_27b","qwen3_4b","mistral_7b",
    "llama31_8b","llama31_70b",
    "gemini2_flash","gemini3_flash","gemini31_pro",
    "gpt54","opus47",
]

def parse_score(val):
    if val is None: return None
    if isinstance(val, list):
        cleaned = [parse_score(v) for v in val]
        cleaned = [c for c in cleaned if c is not None]
        return float(np.mean(cleaned)) if cleaned else None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    m = re.search(r"^\s*(-?\d+\.?\d*)", s)
    if m: return float(m.group(1))
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[-1]) if nums else None


def range_file_stem(rmin, rmax):
    def fmt(v): return str(int(v)) if v == int(v) else str(v)
    return f"range_{fmt(rmin)}_{fmt(rmax)}"


def compute_correlation_for_judge(judge, dataset, metric, rmin, rmax):

    base    = os.path.join(SCORES_DIR, judge, dataset, metric)
    stem    = range_file_stem(rmin, rmax)
    ds_type = DATASETS[dataset]["ds_type"]
    dim_key = DATASETS[dataset]["metrics"][metric]

    group_pred  = defaultdict(list)
    group_human = defaultdict(list)

    def _ingest(fp):
        if not os.path.exists(fp): return
        try:
            data = json.load(open(fp))
        except Exception:
            return
        if ds_type in ("summeval", "newsroom"):
            for item in data:
                s = parse_score(item.get("all_responses"))
                if s is None: continue
                gid = item.get("doc_id", item.get("id"))
                h   = item.get("scores", {}).get(dim_key) or item.get(dim_key)
                if h is None: continue
                if isinstance(h, list): h = float(np.mean(h))
                group_pred[gid].append(float(s))
                group_human[gid].append(float(h))
        elif ds_type == "tc":
            for item in data:
                fact = item.get("fact")
                for resp in item.get("responses", []):
                    s = parse_score(resp.get("all_responses"))
                    if s is None: continue
                    h = resp.get(dim_key)
                    if isinstance(h, list): h = float(np.mean(h))
                    if h is None: continue
                    group_pred[fact].append(float(s))
                    group_human[fact].append(float(h))
        elif ds_type == "wp_a":
            items = list(data.values()) if isinstance(data, dict) else data
            for idx, item in enumerate(items):
                s = parse_score(item.get("all_responses"))
                if s is None: continue
                h = item.get(dim_key)
                if isinstance(h, list): h = float(np.mean(h))
                if h is None: continue
                group_pred[idx].append(float(s))
                group_human[idx].append(float(h))

    for run in range(1, 4):
        _ingest(os.path.join(base, f"{stem}_run{run}.json"))
    if not group_pred:
        _ingest(os.path.join(base, f"{stem}.json"))

    if not group_pred:
        return None

    # wp_a: correlation not per prompt; instead it is over the dataset
    if ds_type == "wp_a":
        pred = [np.mean(v) for v in group_pred.values()]
        hum  = [np.mean(v) for v in group_human.values()]
        if len(pred) < 4 or len(set(hum)) < 2 or len(set(pred)) < 2:
            return None
        return {
            "pearson":    float(pearsonr(pred, hum)[0]),
            "spearman":   float(spearmanr(pred, hum)[0]),
            "kendalltau": float(kendalltau(pred, hum)[0]),
            "n": len(pred),
        }

    # summeval / newsroom / tc: average per-group correlation
    pear_sum = spear_sum = kend_sum = n = 0
    for gid in group_pred:
        p = group_pred[gid]
        h = group_human[gid]
        mn = min(len(p), len(h))
        p, h = p[:mn], h[:mn]
        if mn < 3 or len(set(h)) < 2 or len(set(p)) < 2: continue
        pear_sum  += pearsonr(p, h)[0]
        spear_sum += spearmanr(p, h)[0]
        kend_sum  += kendalltau(p, h)[0]
        n += 1

    if n == 0: return None
    return {
        "pearson":    pear_sum  / n,
        "spearman":   spear_sum / n,
        "kendalltau": kend_sum  / n,
        "n": n,
    }

def main():
    out_fp = os.path.join(REPORTS, "standard_range_correlations.csv")
    print("Computing correlations ...")
    print(f"Output -> {out_fp}\n")

    rows = []
    for judge in ALL_JUDGES:
        for dataset, ds_cfg in DATASETS.items():
            for metric in ds_cfg["metrics"]:
                can = CANONICAL[dataset][metric]
                for r in STD_RANGES:
                    rmin, rmax = r
                    res = compute_correlation_for_judge(judge, dataset, metric, rmin, rmax)
                    if res is None:
                        continue
                    rows.append({
                        "model":      judge,
                        "dataset":    dataset,
                        "metric":     metric,
                        "rmin":       rmin,
                        "rmax":       rmax,
                        "is_canon":   (r == can),
                        "pearson":    round(res["pearson"],    6),
                        "spearman":   round(res["spearman"],   6),
                        "kendalltau": round(res["kendalltau"], 6),
                        "n_corr":     res["n"],
                    })
        print(f"  {judge}: done")

    df = pd.DataFrame(rows)
    df.to_csv(out_fp, index=False)
    print(f"\nSaved {len(df)} rows -> {out_fp}")


if __name__ == "__main__":
    main()
