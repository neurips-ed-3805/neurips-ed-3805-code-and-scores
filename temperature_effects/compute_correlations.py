import os
import re
import csv
import json
from scipy.stats import spearmanr, pearsonr, kendalltau

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES  = os.path.join(_ROOT, "scores", "temperature_effects")
REPORTS = os.path.join(_ROOT, "reports", "temperature_effects")

STANDARD_RANGES = [(0, 1), (1, 3), (1, 5), (0, 10), (0, 100)]

# variant -> runs to average (None = single file range_X_Y.json; ints = range_X_Y_runN.json)
VARIANTS = {
    "gemini3_flash_t0":   [None],
    "gemini31_pro_t0":    [None],
    "gemma2_9b_default":  [1, 2, 3],
    "llama31_8b_default": [1, 2, 3],
}

DATASETS = {
    "summeval": {"dataset_type": "summeval", "metrics": {
        "coh": ("coherence", (1, 5)), "con": ("consistency", (1, 5)),
        "flu": ("fluency", (1, 5)),   "rel": ("relevance", (1, 5))}},
    "newsroom": {"dataset_type": "newsroom", "metrics": {
        "coh": ("coherence", (1, 5)), "flu": ("fluency", (1, 5)),
        "inf": ("informativeness", (1, 5)), "rel": ("relevance", (1, 5))}},
    "tc": {"dataset_type": "tc", "metrics": {
        "coh": ("Understandable", (0, 1)), "eng": ("Engaging", (1, 3)),
        "gro": ("Uses Knowledge", (0, 1)), "nat": ("Natural", (1, 3))}},
    "wp_a": {"dataset_type": "wp_a", "metrics": {
        "coh": ("cohesive", (1, 5)), "enj": ("enjoy", (1, 5)),
        "gra": ("grammar", (1, 5)),  "rel": ("relevant", (1, 5))}},
}


def parse_output(output):

    if isinstance(output, (int, float)):
        return float(output)
    if not isinstance(output, str):
        return None
    matched = re.search(r"^\s*(-?\d+\.?\d*)", output)
    if matched:
        try:
            return float(matched.group(1))
        except Exception:
            pass
    all_nums = re.findall(r"-?\d+\.?\d*", output)
    if all_nums:
        try:
            return float(all_nums[-1])
        except Exception:
            pass
    return None


def safe_avg(scores):
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None


def corr_summeval(jobj, dimension):
    pred, hum = {}, {}
    for item in jobj:
        d = item["doc_id"]
        pred.setdefault(d, []); hum.setdefault(d, [])
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        if s is None:
            continue
        pred[d].append(s)
        hum[d].append(item["scores"][dimension])
    acc = {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    n = 0
    for d in pred:
        p, h = pred[d], hum[d]
        if len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        acc["pearson"] += pearsonr(p, h)[0]
        acc["spearman"] += spearmanr(p, h)[0]
        acc["kendalltau"] += kendalltau(p, h)[0]
        n += 1
    if n == 0:
        return {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    return {k: v / n for k, v in acc.items()}


def corr_tc(jobj, dimension):
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
    acc = {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    n = 0
    for d in pred:
        p, h = pred[d], hum[d]
        if len(set(h)) <= 1 or len(set(p)) <= 1:
            continue
        acc["pearson"] += pearsonr(p, h)[0]
        acc["spearman"] += spearmanr(p, h)[0]
        acc["kendalltau"] += kendalltau(p, h)[0]
        n += 1
    if n == 0:
        return {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    return {k: v / n for k, v in acc.items()}


def corr_wp_a(jobj, dimension):
    if isinstance(jobj, dict):
        jobj = list(jobj.values())
    pred, hum = [], []
    for item in jobj:
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        if s is None:
            continue
        pred.append(s)
        h = item[dimension]
        hum.append(sum(h) / len(h) if isinstance(h, list) else h)
    if len(set(pred)) <= 1 or len(set(hum)) <= 1:
        return {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    return {
        "pearson": pearsonr(pred, hum)[0],
        "spearman": spearmanr(pred, hum)[0],
        "kendalltau": kendalltau(pred, hum)[0],
    }


_CORR = {"summeval": corr_summeval, "newsroom": corr_summeval, "tc": corr_tc, "wp_a": corr_wp_a}


def range_category(rmin, rmax, canonical):
    if (rmin, rmax) == canonical:
        return "canonical"
    if (rmin, rmax) in STANDARD_RANGES:
        return "standard"
    return "others"


def evaluate_one(variant, dataset, metric, dimension, rmin, rmax, run):
    stem = f"range_{rmin}_{rmax}" + ("" if run is None else f"_run{run}")
    fp = os.path.join(SCORES, variant, dataset, metric, stem + ".json")
    if not os.path.exists(fp):
        return None
    return _CORR[DATASETS[dataset]["dataset_type"]](json.load(open(fp)), dimension)


def main():
    for variant, runs in VARIANTS.items():
        for ds, cfg in DATASETS.items():
            rows = []
            for metric, (dimension, canonical) in cfg["metrics"].items():
                for (rmin, rmax) in STANDARD_RANGES:
                    run_results = [r for r in
                                   (evaluate_one(variant, ds, metric, dimension, rmin, rmax, run) for run in runs)
                                   if r is not None]
                    if not run_results:
                        continue
                    n = len(run_results)
                    rows.append({
                        "metric": metric, "range_min": rmin, "range_max": rmax,
                        "range_width": rmax - rmin,
                        "range_category": range_category(rmin, rmax, canonical),
                        "is_canonical": (rmin, rmax) == canonical,
                        "pearson": sum(r["pearson"] for r in run_results) / n,
                        "spearman": sum(r["spearman"] for r in run_results) / n,
                        "kendalltau": sum(r["kendalltau"] for r in run_results) / n,
                        "n_runs": n,
                    })
            if not rows:
                continue
            out_dir = os.path.join(REPORTS, variant, ds)
            os.makedirs(out_dir, exist_ok=True)
            fp = os.path.join(out_dir, "correlations.csv")
            with open(fp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["metric", "range_min", "range_max", "range_width",
                                                  "range_category", "is_canonical",
                                                  "pearson", "spearman", "kendalltau", "n_runs"])
                w.writeheader()
                w.writerows(rows)
            print(f"  wrote {len(rows)} rows -> reports/temperature_effects/{variant}/{ds}/correlations.csv")


if __name__ == "__main__":
    main()
