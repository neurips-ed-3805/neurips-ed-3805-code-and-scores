"""
Usage:
  python calibration_sweep.py --judge gpt54
  python calibration_sweep.py --judge gpt54 --B 1000 --seed 42
"""

import os
import json
import csv
import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau
import re

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
SCORES    = os.path.join(_ROOT, "scores")            
FINAL_RES = os.path.join(_ROOT, "reports")           

STANDARD_RANGES = [(0, 1), (1, 3), (1, 5), (0, 10), (0, 100)]

DATASETS = {
    "summeval": {"dataset_type": "summeval", "metrics": {
        "coh": ("_", "coherence", (1, 5)), "con": ("_", "consistency", (1, 5)),
        "flu": ("_", "fluency", (1, 5)),   "rel": ("_", "relevance", (1, 5))}},
    "newsroom": {"dataset_type": "newsroom", "metrics": {
        "coh": ("_", "coherence", (1, 5)), "flu": ("_", "fluency", (1, 5)),
        "inf": ("_", "informativeness", (1, 5)), "rel": ("_", "relevance", (1, 5))}},
    "tc": {"dataset_type": "tc", "metrics": {
        "coh": ("_", "Understandable", (0, 1)), "eng": ("_", "Engaging", (1, 3)),
        "gro": ("_", "Uses Knowledge", (0, 1)), "nat": ("_", "Natural", (1, 3))}},
    "wp_a": {"dataset_type": "wp_a", "metrics": {
        "coh": ("_", "cohesive", (1, 5)), "enj": ("_", "enjoy", (1, 5)),
        "gra": ("_", "grammar", (1, 5)),  "rel": ("_", "relevant", (1, 5))}},
}

def scores_dir(judge, dataset, metric):
    return os.path.join(SCORES, judge, dataset, metric)

def range_str(rmin, rmax):
    return f"range_{rmin}_{rmax}"

def parse_output(output):

    if isinstance(output, (int, float)):
        return float(output)
    if not isinstance(output, str):
        return None
    matched = re.search(r"^\s*(-?\d+\.?\d*)", output)
    if matched:
        try: return float(matched.group(1))
        except: pass
    all_nums = re.findall(r"-?\d+\.?\d*", output)
    if all_nums:
        try: return float(all_nums[-1])
        except: pass
    return None

def safe_avg(scores):
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None


JUDGES = ["gemma2_9b", "gemma3_27b", "qwen3_4b", "mistral_7b", "llama31_8b", "llama31_70b", "gemini2_flash", "gemini3_flash", "gemini31_pro", "gpt54", "opus47",
]
NLG_DATASETS = ["summeval", "newsroom", "tc", "wp_a"]
MULTI_RUN_JUDGES = {"gemini2_flash", "gemini3_flash"}  # 3 runs -> compute-then-average
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_B = 1000
DEFAULT_SEED = 42
STD = list(STANDARD_RANGES)  


def runs_for(judge):
    return [1, 2, 3] if judge in MULTI_RUN_JUDGES else [None]


def load_score_file(judge, dataset, metric, rmin, rmax, run):
    fp = os.path.join(scores_dir(judge, dataset, metric),
                      range_str(rmin, rmax) + ("" if run is None else f"_run{run}") + ".json")
    if not os.path.exists(fp):
        return None
    return json.load(open(fp))

def per_doc_corrs_summeval(jobj, dimension):
\
    pred, hum = {}, {}
    for item in jobj:
        d = item["doc_id"]
        pred.setdefault(d, []); hum.setdefault(d, [])
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        if s is None:
            continue
        pred[d].append(s)
        hum[d].append(item["scores"][dimension])
    out = {}
    for d in pred:
        p, h = pred[d], hum[d]
        if len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        out[d] = (spearmanr(p, h)[0], kendalltau(p, h)[0])
    return out


def per_doc_corrs_tc(jobj, dimension):
    pred, hum = {}, {}
    for item in jobj:
        d = item["fact"]
        pred.setdefault(d, []); hum.setdefault(d, [])
        for resp in item["responses"]:
            s = safe_avg([parse_output(x) for x in resp["all_responses"]])
            if s is None:
                continue
            pred[d].append(s)
            hv = resp[dimension]
            hum[d].append(sum(hv) / len(hv) if isinstance(hv, list) else hv)
    out = {}
    for d in pred:
        p, h = pred[d], hum[d]
        if len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        out[d] = (spearmanr(p, h)[0], kendalltau(p, h)[0])
    return out


def flat_items_wp_a(jobj, dimension):

    if isinstance(jobj, dict):
        jobj = list(jobj.values())
    out = {}
    for i, item in enumerate(jobj):
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        if s is None:
            continue
        h = item[dimension]
        out[i] = (s, sum(h) / len(h) if isinstance(h, list) else h)
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


def summary_subset_corr(per_run_doc, docs, idx):

    run_vals = []
    for doc_map in per_run_doc:
        vals = [doc_map[d][idx] for d in docs if d in doc_map]
        m = _mean(vals)
        if not np.isnan(m):
            run_vals.append(m)
    return _mean(run_vals)


def wp_subset_corr(per_run_items, idxs):

    idxs = set(idxs)
    sp_runs, kt_runs = [], []
    for item_map in per_run_items:
        p = [item_map[i][0] for i in idxs if i in item_map]
        h = [item_map[i][1] for i in idxs if i in item_map]
        if len(set(p)) <= 1 or len(set(h)) <= 1 or len(p) < 3:
            continue
        sp_runs.append(spearmanr(p, h)[0])
        kt_runs.append(kendalltau(p, h)[0])
    return _mean(sp_runs), _mean(kt_runs)


def process_metric(judge, dataset, metric):

    ds_type = DATASETS[dataset]["dataset_type"]
    dimension = DATASETS[dataset]["metrics"][metric][1]
    canonical = tuple(DATASETS[dataset]["metrics"][metric][2])
    runs = runs_for(judge)

    range_data = {}   # (rmin,rmax)
    for (rmin, rmax) in STD:
        per_run = []
        for run in runs:
            jobj = load_score_file(judge, dataset, metric, rmin, rmax, run)
            if jobj is None:
                continue
            if ds_type in ("summeval", "newsroom"):
                per_run.append(per_doc_corrs_summeval(jobj, dimension))
            elif ds_type == "tc":
                per_run.append(per_doc_corrs_tc(jobj, dimension))
            elif ds_type == "wp_a":
                per_run.append(flat_items_wp_a(jobj, dimension))
        if per_run:
            range_data[(rmin, rmax)] = per_run

    if len(range_data) < len(STD):
        missing = [r for r in STD if r not in range_data]
        print(f"  [skip] {judge}/{dataset}/{metric}: missing ranges {missing}")
        return None
    if canonical not in range_data:
        print(f"  [skip] {judge}/{dataset}/{metric}: canonical {canonical} missing")
        return None

    is_summary = ds_type in ("summeval", "newsroom", "tc")

    ds_idx = NLG_DATASETS.index(dataset)
    metric_idx = list(DATASETS[dataset]["metrics"].keys()).index(metric)

    if is_summary:
        universe = set()
        for per_run in range_data.values():
            for doc_map in per_run:
                universe |= set(doc_map.keys())
        universe = sorted(universe)
    else:
        universe = set()
        for per_run in range_data.values():
            for item_map in per_run:
                universe |= set(item_map.keys())
        universe = sorted(universe)
    n = len(universe)
    universe = np.array(universe, dtype=object)

    def corr_on(subset_units, rng_key, idx):
        if is_summary:
            return summary_subset_corr(range_data[rng_key], subset_units, idx)
        else:
            return wp_subset_corr(range_data[rng_key], subset_units)[idx]

    results = {}
    for fi, frac in enumerate(FRACTIONS):
        k = max(1, int(round(frac * n)))
        if k >= n:
            k = n - 1
        rng = np.random.default_rng([DEFAULT_SEED, ds_idx, metric_idx, int(frac * 100)])
        sel_sp, can_sp, sel_kt, can_kt, regret, pick_canon = [], [], [], [], [], []
        pick_counts = {r: 0 for r in STD}
        for b in range(process_metric.B):
            perm = rng.permutation(n)
            calib = universe[perm[:k]]
            test = universe[perm[k:]]

            calib_sp = {r: corr_on(calib, r, 0) for r in STD}
            best_r, best_v = None, -np.inf
            for r in STD:  
                v = calib_sp[r]
                if not np.isnan(v) and v > best_v:
                    best_v, best_r = v, r
            if best_r is None:
                best_r = canonical

            test_sp = {r: corr_on(test, r, 0) for r in STD}
            test_kt = {r: corr_on(test, r, 1) for r in STD}
            s_sel = test_sp[best_r]; s_can = test_sp[canonical]
            k_sel = test_kt[best_r]; k_can = test_kt[canonical]

            if np.isnan(s_sel) or np.isnan(s_can):
                continue

            sel_sp.append(s_sel); can_sp.append(s_can)
            sel_kt.append(k_sel); can_kt.append(k_can)
            oracle = max(v for v in test_sp.values() if not np.isnan(v))
            regret.append(oracle - s_sel)
            pick_canon.append(1.0 if best_r == canonical else 0.0)
            pick_counts[best_r] += 1
            
        nb = len(sel_sp)
        results[f"{frac:.1f}"] = {
            "n_units": n, "calib_size": k, "n_boot": nb,
            "sel_spearman": _mean(sel_sp), "canon_spearman": _mean(can_sp),
            "delta_spearman": _mean(sel_sp) - _mean(can_sp),
            "sel_kendall": _mean(sel_kt), "canon_kendall": _mean(can_kt),
            "sel_spearman_std": float(np.std(sel_sp)) if nb else np.nan,
            "test_oracle_regret": _mean(regret),
            "pick_canonical_rate": _mean(pick_canon),
            "pick_distribution": {f"{r[0]}_{r[1]}": pick_counts[r] / max(1, sum(pick_counts.values())) for r in STD},
            "canonical_range": f"{canonical[0]}_{canonical[1]}",
        }
    return results


def main():
    global DEFAULT_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    DEFAULT_SEED = args.seed
    process_metric.B = args.B

    judge = args.judge
    out_dir = os.path.join(FINAL_RES, "non_oracle_gains", "per_judge")
    os.makedirs(out_dir, exist_ok=True)

    full = {"judge": judge, "B": args.B, "seed": args.seed, "datasets": {}}
    csv_rows = []
    print(f"=== {judge} (B={args.B}) ===")
    for ds in NLG_DATASETS:
        full["datasets"][ds] = {"metrics": {}}
        metrics = list(DATASETS[ds]["metrics"].keys())
        for m in metrics:
            res = process_metric(judge, ds, m)
            if res is None:
                continue
            full["datasets"][ds]["metrics"][m] = res
            for frac, agg in res.items():
                csv_rows.append({
                    "judge": judge, "dataset": ds, "metric": m, "fraction": frac,
                    "sel_spearman": round(agg["sel_spearman"], 4),
                    "canon_spearman": round(agg["canon_spearman"], 4),
                    "delta_spearman": round(agg["delta_spearman"], 4),
                    "test_oracle_regret": round(agg["test_oracle_regret"], 4),
                    "pick_canonical_rate": round(agg["pick_canonical_rate"], 3),
                    "canonical_range": agg["canonical_range"],
                    "calib_size": agg["calib_size"], "n_units": agg["n_units"],
                })
            print(f"  {ds}/{m}: " + " | ".join(
                f"f={frac} sel={r['sel_spearman']:.3f} can={r['canon_spearman']:.3f} d={r['delta_spearman']:+.3f}"
                for frac, r in res.items()))

        # dataset-level macro-average over metrics, per fraction
        ds_avg = {}
        for frac in [f"{x:.1f}" for x in FRACTIONS]:
            sels = [full["datasets"][ds]["metrics"][m][frac]["sel_spearman"]
                    for m in full["datasets"][ds]["metrics"] if frac in full["datasets"][ds]["metrics"][m]]
            cans = [full["datasets"][ds]["metrics"][m][frac]["canon_spearman"]
                    for m in full["datasets"][ds]["metrics"] if frac in full["datasets"][ds]["metrics"][m]]
            regs = [full["datasets"][ds]["metrics"][m][frac]["test_oracle_regret"]
                    for m in full["datasets"][ds]["metrics"] if frac in full["datasets"][ds]["metrics"][m]]
            picks = [full["datasets"][ds]["metrics"][m][frac]["pick_canonical_rate"]
                     for m in full["datasets"][ds]["metrics"] if frac in full["datasets"][ds]["metrics"][m]]
            if sels:
                ds_avg[frac] = {
                    "sel_spearman": _mean(sels), "canon_spearman": _mean(cans),
                    "delta_spearman": _mean(sels) - _mean(cans),
                    "test_oracle_regret": _mean(regs), "pick_canonical_rate": _mean(picks),
                }
        full["datasets"][ds]["average"] = ds_avg

    # overall (macro over the 4 datasets' averages)
    overall = {}
    for frac in [f"{x:.1f}" for x in FRACTIONS]:
        sels = [full["datasets"][ds]["average"][frac]["sel_spearman"]
                for ds in NLG_DATASETS if frac in full["datasets"][ds].get("average", {})]
        cans = [full["datasets"][ds]["average"][frac]["canon_spearman"]
                for ds in NLG_DATASETS if frac in full["datasets"][ds].get("average", {})]
        if sels:
            overall[frac] = {"sel_spearman": _mean(sels), "canon_spearman": _mean(cans),
                             "delta_spearman": _mean(sels) - _mean(cans)}
    full["overall"] = overall

    json.dump(full, open(os.path.join(out_dir, f"{judge}.json"), "w"), indent=2)
    with open(os.path.join(out_dir, f"{judge}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"  -> wrote {out_dir}/{judge}.json and {judge}.csv")
    print("  OVERALL: " + " | ".join(
        f"f={frac} sel={o['sel_spearman']:.3f} can={o['canon_spearman']:.3f} d={o['delta_spearman']:+.3f}"
        for frac, o in overall.items()))


if __name__ == "__main__":
    process_metric.B = DEFAULT_B
    main()
