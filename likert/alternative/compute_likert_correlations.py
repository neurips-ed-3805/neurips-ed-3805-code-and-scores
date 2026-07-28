import os, re, csv, json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau

_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORES  = os.path.join(_ROOT, "scores", "likert")
HERE    = os.path.dirname(os.path.abspath(__file__))

SCALES  = [3, 5, 7]

DATASETS = {
    "summeval": {"metrics": {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"},        "ds_type": "summeval"},
    "newsroom": {"metrics": {"coh": "coherence", "flu": "fluency", "inf": "informativeness", "rel": "relevance"},    "ds_type": "newsroom"},
    "tc":       {"metrics": {"coh": "Understandable", "eng": "Engaging", "gro": "Uses Knowledge", "nat": "Natural"}, "ds_type": "tc"},
    "wp_a":     {"metrics": {"coh": "cohesive", "enj": "enjoy", "gra": "grammar", "rel": "relevant"},                "ds_type": "wp_a"},
}

ALL_JUDGES = [
    "gemma2_9b", "gemma3_27b", "qwen3_4b", "mistral_7b", "llama31_8b", "llama31_70b",
    "gemini2_flash", "gemini3_flash", "gemini31_pro",
]


def parse_score(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        val = val[0] if val else None
        if val is None:
            return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(val))
        return float(m.group()) if m else None


def human_score(h):
    if h is None:
        return None
    if isinstance(h, list):
        return float(np.mean(h)) if h else None
    try:
        return float(h)
    except (ValueError, TypeError):
        return None


def _scale_file(judge, dataset, metric, scale):
    d = os.path.join(SCORES, judge, dataset, metric)
    run1 = os.path.join(d, f"scale_{scale}_run1.json")
    return run1 if os.path.exists(run1) else os.path.join(d, f"scale_{scale}.json")


def compute_corr(judge, dataset, metric, scale):
    cfg     = DATASETS[dataset]
    dim_key = cfg["metrics"][metric]
    ds_type = cfg["ds_type"]
    fpath   = _scale_file(judge, dataset, metric, scale)
    if not os.path.exists(fpath):
        return None
    try:
        data = json.load(open(fpath))
    except Exception:
        return None

    gh, gj = defaultdict(list), defaultdict(list)
    if ds_type in ("summeval", "newsroom"):
        for item in data:
            s = parse_score(item.get("all_responses"))
            h = item.get("scores", {}).get(dim_key) if isinstance(item.get("scores"), dict) else None
            if h is None:
                h = item.get(dim_key)
            h = human_score(h)
            if s is not None and h is not None:
                gid = item.get("doc_id", item.get("id"))
                gh[gid].append(h); gj[gid].append(s)
    elif ds_type == "tc":
        for item in data:
            gid = item.get("fact")
            for resp in item.get("responses", []):
                s = parse_score(resp.get("all_responses"))
                h = human_score(resp.get(dim_key))
                if s is not None and h is not None:
                    gh[gid].append(h); gj[gid].append(s)
    elif ds_type == "wp_a":
        items = list(data.values()) if isinstance(data, dict) else data
        for item in items:
            s = parse_score(item.get("all_responses"))
            h = human_score(item.get(dim_key))
            if s is not None and h is not None:
                gh["__all__"].append(h); gj["__all__"].append(s)

    if ds_type == "wp_a":
        h, j = gh["__all__"], gj["__all__"]
        if len(h) < 4 or len(set(h)) < 2 or len(set(j)) < 2:
            return None
        return {"spearman": float(spearmanr(h, j)[0]), "kendall": float(kendalltau(h, j)[0]), "n": len(h)}

    sp = kt = 0.0
    ng = 0
    for gid in gh:
        h, j = gh[gid], gj[gid]
        if len(h) < 3 or len(set(h)) < 2 or len(set(j)) < 2:
            continue
        sp += spearmanr(h, j)[0]; kt += kendalltau(h, j)[0]; ng += 1
    if ng == 0:
        return None
    return {"spearman": sp / ng, "kendall": kt / ng, "n": ng}


def main():
    if not os.path.isdir(SCORES):
        raise SystemExit(f"No Likert scores at {SCORES} -- run scores/extract_all.sh first.")
    present = {d for d in os.listdir(SCORES) if os.path.isdir(os.path.join(SCORES, d))}
    judges  = [j for j in ALL_JUDGES if j in present]

    rows = []
    for judge in judges:
        for ds, cfg in DATASETS.items():
            for metric in cfg["metrics"]:
                for scale in SCALES:
                    c = compute_corr(judge, ds, metric, scale)
                    if c:
                        rows.append([judge, ds, metric, scale,
                                     f"{c['spearman']:.6f}", f"{c['kendall']:.6f}", c["n"]])

    out = os.path.join(HERE, "likert_correlations.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "dataset", "metric", "scale", "spearman", "kendall", "n"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
