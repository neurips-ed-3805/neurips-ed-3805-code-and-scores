"""
BiGGen-Bench meta-evaluation.
"""
import os
import re
import csv
import json
from scipy.stats import spearmanr, pearsonr, kendalltau

_HERE   = os.path.dirname(os.path.abspath(__file__))
SCORES  = os.path.join(_HERE, "scores")
REPORTS = os.path.join(_HERE, "reports")

DIMENSION = "quality"         
CANONICAL = (1, 5)             
STANDARD_RANGES = {(0, 1), (1, 3), (1, 5), (0, 10), (0, 100)}

RANGES = [(0, 1), (1, 3), (1, 5), (1, 10), (0, 10), (0, 100),
          (0, 0.5), (0, 2), (0, 5), (5, 10), (20, 25),
          (-0.1, 0.1), (-0.5, 0.5), (-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5),
          (-1, 3), (0, 1000), (0, 4)]

JUDGES = ["gemini31_pro", "gemini3_flash", "gemma2_9b", "gemma3_27b", "llama31_70b", "llama31_8b", "mistral_7b", "qwen3_4b"]
API = {"gemini31_pro", "gemini3_flash"}   # these store results as range_X_Y_run1.json


def parse_output(output):
    if isinstance(output, (int, float)):
        return float(output)
    if not isinstance(output, str):
        return None
    
    m = re.search(r"^\s*(-?\d+\.?\d*)", output)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    nums = re.findall(r"-?\d+\.?\d*", output)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            pass
    return None


def safe_avg(scores):
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else None


def flat_corr(jobj):

    pred, hum = [], []

    for item in jobj:
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        if s is None:
            continue

        h = item.get("scores", {}).get(DIMENSION)
        if h is None:
            continue
        pred.append(s)
        hum.append(h)

    if len(set(pred)) <= 1 or len(set(hum)) <= 1:
        return None
    
    return {"pearson": pearsonr(pred, hum)[0],
            "spearman": spearmanr(pred, hum)[0],
            "kendalltau": kendalltau(pred, hum)[0]}


def range_str(rmin, rmax):
    return f"range_{rmin}_{rmax}"


def evaluate(judge, rmin, rmax):

    base = os.path.join(SCORES, judge, "biggen", "qua")
    stem = range_str(rmin, rmax)
    results = []

    if judge in API:
        for run in (1, 2, 3):
            fp = os.path.join(base, f"{stem}_run{run}.json")
            if os.path.exists(fp):
                r = flat_corr(json.load(open(fp)))
                if r:
                    results.append(r)
    if not results:
        fp = os.path.join(base, f"{stem}.json")
        if os.path.exists(fp):
            r = flat_corr(json.load(open(fp)))
            if r:
                results.append(r)

    if not results:
        return None
    n = len(results)
    return {k: sum(r[k] for r in results) / n for k in ("pearson", "spearman", "kendalltau")}, n


def range_category(rmin, rmax):
    if (rmin, rmax) == CANONICAL:
        return "canonical"
    if (rmin, rmax) in STANDARD_RANGES:
        return "standard"
    return "others"


def main():
    for judge in JUDGES:
        rows = []
        for (rmin, rmax) in RANGES:
            res = evaluate(judge, rmin, rmax)
            if res is None:
                continue
            corr, n = res

            rows.append({
                "metric": "qua", "range_min": rmin, "range_max": rmax,
                "range_width": rmax - rmin,
                "range_category": range_category(rmin, rmax),
                "is_canonical": (rmin, rmax) == CANONICAL,
                "pearson": corr["pearson"], "spearman": corr["spearman"],
                "kendalltau": corr["kendalltau"], "n_runs": n,
            })

        if not rows:
            continue

        out_dir = os.path.join(REPORTS, judge)
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, "correlations.csv")

        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "range_min", "range_max", "range_width", "range_category", "is_canonical", "pearson", "spearman", "kendalltau", "n_runs"])

            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} rows -> reports/{judge}/correlations.csv")


if __name__ == "__main__":
    main()
