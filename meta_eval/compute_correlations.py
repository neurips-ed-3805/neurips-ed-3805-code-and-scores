"""
Per-judge / per-range correlation values (all 20 ranges) -> done for non frontier judges

Spearman/Kendall are averaged over the 3 runs (run1/2/3) for the multi-run API judges (gemini2_flash, gemini3_flash); the open judges have a single file per range.
"""

import json, os, re, csv
import numpy as np
from scipy.stats import spearmanr, kendalltau

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
SCORES  = os.path.join(_ROOT, "scores")
REPORTS = os.path.join(_ROOT, "reports")


RANGES = [(0, 1), (1, 3), (1, 5), (1, 10), (0, 10), (0, 100),
    (0, 0.5), (0, 2), (0, 5), (5, 10), (20, 25),
    (-0.1, 0.1), (-0.5, 0.5), (-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-1, 3), (0, 1000),]


MULTI_RUN_JUDGES = {"gemini2_flash", "gemini3_flash"}

JUDGES = ["gemini3_flash", "gemini2_flash",
    "llama31_70b", "gemma3_27b", "gemma2_9b",
    "llama31_8b", "mistral_7b", "qwen3_4b",]

DATASETS = {
    "summeval": {"metrics": {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"},        "dataset_type": "summeval"},
    "newsroom": {"metrics": {"coh": "coherence", "flu": "fluency", "inf": "informativeness", "rel": "relevance"},    "dataset_type": "newsroom"},
    "tc":       {"metrics": {"coh": "Understandable", "eng": "Engaging", "gro": "Uses Knowledge", "nat": "Natural"}, "dataset_type": "tc"},
    "wp_a":     {"metrics": {"coh": "cohesive", "enj": "enjoy", "gra": "grammar", "rel": "relevant"},                "dataset_type": "wp_a"},
}

def range_str(rmin, rmax):
    return f"range_{rmin}_{rmax}"

def scores_dir(judge, dataset, metric_short):
    return os.path.join(SCORES, judge, dataset, metric_short)

def parse_output(output):
    if isinstance(output, (int, float)):
        return float(output)
    if not isinstance(output, str):
        return None
    m = re.search(r"^\s*(-?\d+\.?\d*)", output)
    if m:
        try: return float(m.group(1))
        except: pass
    all_nums = re.findall(r"-?\d+\.?\d*", output)
    if all_nums:
        try: return float(all_nums[-1])
        except: pass
    return None

def safe_avg(scores):
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None

def compute_corr_summeval(jobj, dimension):
    """Group by doc_id, compute per-doc correlations, average."""
    pred_scores, human_scores = {}, {}
    for item in jobj:
        doc_id = item["doc_id"]
        if doc_id not in pred_scores:
            pred_scores[doc_id] = []
            human_scores[doc_id] = []
        all_scores = [parse_output(x) for x in item.get("all_responses", [])]
        score = safe_avg(all_scores)
        if score is None:
            continue
        pred_scores[doc_id].append(score)
        human_scores[doc_id].append(item["scores"][dimension])

    rho_sum, tau_sum, ctr = 0.0, 0.0, 0
    for doc_id in pred_scores:
        p = pred_scores[doc_id]
        h = human_scores[doc_id]
        if len(p) < 2 or len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        rho_sum += spearmanr(p, h)[0]
        tau_sum += kendalltau(p, h)[0]
        ctr += 1
    if ctr == 0:
        return None, None
    return rho_sum / ctr, tau_sum / ctr

def compute_corr_tc(jobj, dimension):
    """Group by fact, compute per-fact correlations, average."""
    pred_scores, human_scores = {}, {}
    for item in jobj:
        doc_id = item["fact"]
        if doc_id not in pred_scores:
            pred_scores[doc_id] = []
            human_scores[doc_id] = []
        for resp in item.get("responses", []):
            all_scores = [parse_output(x) for x in resp.get("all_responses", [])]
            score = safe_avg(all_scores)
            if score is None:
                continue
            pred_scores[doc_id].append(score)
            human_vals = resp[dimension]
            human_scores[doc_id].append(sum(human_vals) / len(human_vals))

    rho_sum, tau_sum, ctr = 0.0, 0.0, 0
    for doc_id in pred_scores:
        p = pred_scores[doc_id]
        h = human_scores[doc_id]
        if len(p) < 2 or len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        rho_sum += spearmanr(p, h)[0]
        tau_sum += kendalltau(p, h)[0]
        ctr += 1
    if ctr == 0:
        return None, None
    return rho_sum / ctr, tau_sum / ctr

def compute_corr_wp_a(jobj, dimension):
    """Flat list, one correlation across all items."""
    if isinstance(jobj, dict):
        jobj = list(jobj.values())
    pred_scores, human_scores = [], []
    for item in jobj:
        all_scores = [parse_output(x) for x in item.get("all_responses", [])]
        score = safe_avg(all_scores)
        if score is None:
            continue
        pred_scores.append(score)
        h = item[dimension]
        human_scores.append(sum(h) / len(h) if isinstance(h, list) else h)

    if len(pred_scores) < 2 or len(set(pred_scores)) <= 1 or len(set(human_scores)) <= 1:
        return None, None
    return spearmanr(pred_scores, human_scores)[0], kendalltau(pred_scores, human_scores)[0]

def evaluate_file(dataset, metric_short, fp):
    """Evaluate one score file, returns (rho, tau) or (None, None)."""
    ds_cfg = DATASETS[dataset]
    dimension = ds_cfg["metrics"][metric_short]
    if not os.path.exists(fp):
        return None, None
    try:
        jobj = json.load(open(fp))
    except Exception:
        return None, None

    ds_type = ds_cfg["dataset_type"]
    if ds_type in ("summeval", "newsroom"):
        return compute_corr_summeval(jobj, dimension)
    elif ds_type == "tc":
        return compute_corr_tc(jobj, dimension)
    elif ds_type == "wp_a":
        return compute_corr_wp_a(jobj, dimension)
    return None, None

def compute_avg_corr(judge, dataset, metric_short, rmin, rmax):
    rhos, taus = [], []
    base_dir = scores_dir(judge, dataset, metric_short)
    rs = range_str(rmin, rmax)

    candidate_files = []
    if judge in MULTI_RUN_JUDGES:
        candidate_files.extend([os.path.join(base_dir, f"{rs}_run{run}.json") for run in range(1, 4)])
        candidate_files.append(os.path.join(base_dir, f"{rs}.json"))
    else:
        candidate_files.append(os.path.join(base_dir, f"{rs}.json"))

    for fp in candidate_files:
        rho, tau = evaluate_file(dataset, metric_short, fp)
        if rho is not None:
            rhos.append(rho)
            taus.append(tau)
    if not rhos:
        return None, None
    return np.mean(rhos), np.mean(taus)

def main():
    os.makedirs(REPORTS, exist_ok=True)
    csv_path = os.path.join(REPORTS, "all_range_correlations.csv")
    print(f"Computing correlations from {SCORES} (this may take a few minutes)...")

    results = {}
    for judge in JUDGES:
        results[judge] = {}
        for dataset, ds_cfg in DATASETS.items():
            results[judge][dataset] = {}
            for metric_short in ds_cfg["metrics"]:
                results[judge][dataset][metric_short] = {}
                for rmin, rmax in RANGES:
                    results[judge][dataset][metric_short][(rmin, rmax)] = \
                        compute_avg_corr(judge, dataset, metric_short, rmin, rmax)
                print(f"  {judge}/{dataset}/{metric_short} done")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "dataset", "metric", "range_min", "range_max",
                         "spearman_rho", "kendall_tau"])
        for judge in JUDGES:
            for dataset in DATASETS:
                for metric_short in DATASETS[dataset]["metrics"]:
                    for rmin, rmax in RANGES:
                        rho, tau = results[judge][dataset][metric_short].get((rmin, rmax), (None, None))
                        writer.writerow([judge, dataset, metric_short, rmin, rmax,
                                         f"{rho:.6f}" if rho is not None else "0.000000",
                                         f"{tau:.6f}" if tau is not None else "0.000000"])
    print(f"Wrote correlation values to {csv_path}")

if __name__ == "__main__":
    main()
