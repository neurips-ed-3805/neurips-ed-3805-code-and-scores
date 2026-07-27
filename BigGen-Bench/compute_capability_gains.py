import os
import re
import csv
import json
from scipy.stats import spearmanr, kendalltau

_HERE   = os.path.dirname(os.path.abspath(__file__))
SCORES  = os.path.join(_HERE, "scores")
REPORTS = os.path.join(_HERE, "reports")

DIMENSION = "quality"
CANONICAL = (1, 5)
STANDARD_RANGES = [(0, 1), (1, 3), (1, 5), (0, 10), (0, 100)]
API = {"gemini31_pro", "gemini3_flash"}   # these store results as range_X_Y_run1.json

JUDGES = [("gemini31_pro", "Gemini-3.1-Pro"), ("gemini3_flash", "Gemini-3-Flash"), ("gemma3_27b", "Gemma-3-27B"), ("llama31_70b", "LLaMA-3.1-70B"), ("qwen3_4b", "Qwen3-4B"), ("gemma2_9b", "Gemma-2-9B"), ("llama31_8b", "LLaMA-3.1-8B"), ("mistral_7b", "Mistral-7B"),]

CAPABILITIES = ["grounding", "instruction_following", "planning", "reasoning", "refinement", "safety", "theory_of_mind", "tool_usage"]


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


def load(judge, rmin, rmax):
    base = os.path.join(SCORES, judge, "biggen", "qua")
    stem = f"range_{rmin}_{rmax}"

    for cand in ([f"{stem}_run1.json", f"{stem}.json"] if judge in API else [f"{stem}.json"]):
        fp = os.path.join(base, cand)
        if os.path.exists(fp):
            return json.load(open(fp))
        
    return None


def capability_corrs(jobj):
    buckets = {c: ([], []) for c in CAPABILITIES}
    for item in jobj:
        cap = item.get("capability")
        if cap not in buckets:
            continue
        s = safe_avg([parse_output(x) for x in item["all_responses"]])
        h = item.get("scores", {}).get(DIMENSION)
        if s is None or h is None:
            continue
        buckets[cap][0].append(s)
        buckets[cap][1].append(h)
    out = {}
    for cap, (pred, hum) in buckets.items():
        if len(set(pred)) > 1 and len(set(hum)) > 1:
            out[cap] = (spearmanr(pred, hum)[0], kendalltau(pred, hum)[0])
    return out


def rlabel(r):
    f = lambda v: str(int(v)) if float(v) == int(float(v)) else str(v)
    return f"[{f(r[0])},{f(r[1])}]"


def write_matrix(path, matrix):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge"] + CAPABILITIES)
        for _, jdisp in JUDGES:
            w.writerow([jdisp] + [f"{matrix[(jdisp, c)]:+.3f}" if (jdisp, c) in matrix else "" for c in CAPABILITIES])


def main():
    os.makedirs(REPORTS, exist_ok=True)
    long_rows = []
    sp_matrix, kt_matrix = {}, {}   

    for jdir, jdisp in JUDGES:
        per_range = {r: capability_corrs(load(jdir, *r)) for r in STANDARD_RANGES}

        for cap in CAPABILITIES:
            if cap not in per_range[CANONICAL]:
                continue

            canon_sp, canon_kt = per_range[CANONICAL][cap]
            sp = [(per_range[r][cap][0], r) for r in STANDARD_RANGES if cap in per_range[r]]
            kt = [(per_range[r][cap][1], r) for r in STANDARD_RANGES if cap in per_range[r]]

            best_sp, best_sp_r = max(sp)
            best_kt, best_kt_r = max(kt)
            gain_sp = best_sp - canon_sp
            gain_kt = best_kt - canon_kt

            long_rows.append({
                "judge": jdisp, "capability": cap,
                "canonical_spearman": f"{canon_sp:.4f}",
                "best_std_range_spearman": rlabel(best_sp_r),
                "best_std_spearman": f"{best_sp:.4f}",
                "gain_spearman": f"{gain_sp:+.3f}",
                "canonical_kendall": f"{canon_kt:.4f}",
                "best_std_range_kendall": rlabel(best_kt_r),
                "best_std_kendall": f"{best_kt:.4f}",
                "gain_kendall": f"{gain_kt:+.3f}",
            })

            sp_matrix[(jdisp, cap)] = gain_sp
            kt_matrix[(jdisp, cap)] = gain_kt

    with open(os.path.join(REPORTS, "capability_gains.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "judge", "capability", "canonical_spearman", "best_std_range_spearman",
            "best_std_spearman", "gain_spearman", "canonical_kendall",
            "best_std_range_kendall", "best_std_kendall", "gain_kendall"])
        w.writeheader()
        w.writerows(long_rows)

    write_matrix(os.path.join(REPORTS, "capability_gains_matrix_spearman.csv"), sp_matrix)
    write_matrix(os.path.join(REPORTS, "capability_gains_matrix_kendall.csv"), kt_matrix)
    
    print(f"wrote capability_gains.csv ({len(long_rows)} rows) + spearman/kendall gain matrices to {REPORTS}")


if __name__ == "__main__":
    main()
