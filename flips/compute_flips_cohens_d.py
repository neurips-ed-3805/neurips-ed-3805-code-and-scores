"""
Bilateral-flip + Cohen's-d instability analysis
"""

import os, json, re, warnings
from collections import defaultdict
from itertools import combinations
import csv as _csv

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
SCORES_DIR  = os.path.join(ROOT, "scores")     
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

STD_RANGES = [(0, 1), (1, 3), (1, 5), (0, 10), (0, 100)]

CANONICAL = {
    "summeval": {"coh": (1, 5), "con": (1, 5), "flu": (1, 5), "rel": (1, 5)},
    "newsroom": {"coh": (1, 5), "flu": (1, 5), "inf": (1, 5), "rel": (1, 5)},
    "tc":       {"coh": (0, 1), "eng": (1, 3), "gro": (0, 1), "nat": (1, 3)},
    "wp_a":     {"coh": (1, 5), "enj": (1, 5), "gra": (1, 5), "rel": (1, 5)},
}

DATASETS = {
    "summeval": {
        "ds_type": "summeval",
        "metrics": {"coh": "coherence", "con": "consistency",
                    "flu": "fluency",   "rel": "relevance"},
    },
    "newsroom": {
        "ds_type": "newsroom",
        "metrics": {"coh": "coherence", "flu": "fluency",
                    "inf": "informativeness", "rel": "relevance"},
    },
    "tc": {
        "ds_type": "tc",
        "metrics": {"coh": "Understandable", "eng": "Engaging",
                    "gro": "Uses Knowledge", "nat": "Natural"},
    },
    "wp_a": {
        "ds_type": "wp_a",
        "metrics": {"coh": "cohesive", "enj": "enjoy",
                    "gra": "grammar",  "rel": "relevant"},
    },
}

ALL_JUDGES = [
    "gemma2_9b",    "gemma3_27b",    "qwen3_4b",     "mistral_7b",
    "llama31_8b",   "llama31_70b",
    "gemini2_flash","gemini3_flash", "gemini31_pro",
    "gpt54",        "opus47",
]

JUDGE_LABEL = {
    "gemma2_9b":     "Gemma-2-9B",
    "gemma3_27b":    "Gemma-3-27B",
    "qwen3_4b":      "Qwen3-4B",
    "mistral_7b":    "Mistral-7B",
    "llama31_8b":    "LLaMA-3.1-8B",
    "llama31_70b":   "LLaMA-3.1-70B",
    "gemini2_flash": "Gemini-2.0-Flash",
    "gemini3_flash": "Gemini-3-Flash",
    "gemini31_pro":  "Gemini-3.1-Pro",
    "gpt54":         "GPT-5.4",
    "opus47":        "Claude Opus 4.7",
}

DATASET_LABEL = {
    "summeval": "SummEval",
    "newsroom": "Newsroom",
    "tc":       "Topical Chat",
    "wp_a":     "WP-A",
}

ALPHA = 0.05

def parse_score(val):
    if val is None:
        return None
    if isinstance(val, list):
        cleaned = [parse_score(v) for v in val]
        cleaned = [c for c in cleaned if c is not None]
        return float(np.mean(cleaned)) if cleaned else None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = re.search(r"^\s*(-?\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[-1]) if nums else None


def range_file_stem(rmin, rmax):
    def fmt(v):
        return str(int(v)) if float(v) == int(v) else str(v)
    return f"range_{fmt(rmin)}_{fmt(rmax)}"


def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def load_system_scores(judge, dataset, metric, rmin, rmax):

    base    = os.path.join(SCORES_DIR, judge, dataset, metric)
    stem    = range_file_stem(rmin, rmax)
    ds_type = DATASETS[dataset]["ds_type"]

    acc = defaultdict(list)   # (sys_id, doc_key) -> [scores across runs]

    def _ingest(fp):
        if not os.path.exists(fp):
            return
        try:
            with open(fp) as fh:
                data = json.load(fh)
        except Exception:
            return

        if ds_type == "summeval":
            for item in data:
                s = parse_score(item.get("all_responses"))
                if s is None:
                    continue
                sys_id = item.get("system_id", "unknown")
                doc_id = item.get("doc_id", "?")
                acc[(sys_id, doc_id)].append(s)

        elif ds_type == "newsroom":
            doc_buckets = defaultdict(list)
            for item in data:
                doc_id = item.get("doc_id", "?")
                s = parse_score(item.get("all_responses"))
                doc_buckets[doc_id].append(s)
            for doc_id, score_list in doc_buckets.items():
                for pos, s in enumerate(score_list):
                    if s is None:
                        continue
                    acc[(f"sys{pos}", doc_id)].append(s)

        elif ds_type == "tc":
            for item in data:
                fact = item.get("fact", "?")
                for resp in item.get("responses", []):
                    s = parse_score(resp.get("all_responses"))
                    if s is None:
                        continue
                    model = resp.get("model", "unknown")
                    acc[(model, fact)].append(s)

        elif ds_type == "wp_a":
            items = list(data.values()) if isinstance(data, dict) else data
            # human[i] is paired with ai[i]: same writing prompt.
            human_idx = 0
            ai_idx    = 0
            for item in items:
                s = parse_score(item.get("all_responses"))
                if item.get("human_written", 0) == 1:
                    if s is not None:
                        acc[("human", human_idx)].append(s)
                    human_idx += 1
                else:
                    if s is not None:
                        acc[("ai", ai_idx)].append(s)
                    ai_idx += 1

    for run in range(1, 4):
        _ingest(os.path.join(base, f"{stem}_run{run}.json"))
    if not acc:
        _ingest(os.path.join(base, f"{stem}.json"))

    if not acc:
        return {}

    sys_doc_mean = {k: float(np.mean(v)) for k, v in acc.items()}

    sys_to_docs = defaultdict(dict)
    for (sys_id, doc_key), score in sys_doc_mean.items():
        sys_to_docs[sys_id][doc_key] = score

    all_sys = list(sys_to_docs.keys())
    if len(all_sys) < 2:
        return {}

    # Intersect doc-keys across all systems so lists are aligned for paired tests
    doc_sets    = [set(sys_to_docs[s].keys()) for s in all_sys]
    common_docs = sorted(doc_sets[0].intersection(*doc_sets[1:]))

    if len(common_docs) < 3:
        return {}

    return {sys_id: [sys_to_docs[sys_id][d] for d in common_docs]
            for sys_id in all_sys}

def compare_pair(scores_x, scores_y):
    """
    Run all three significance tests on the aligned score vectors (scores_x, scores_y).

    Returns a dict with keys:
      direction : +1 if mean(X) > mean(Y), -1 if mean(X) < mean(Y), 0 if equal
      pval_w    : p-value from Wilcoxon signed-rank (two-sided; primary)
      pval_t1   : p-value from t-test 1 (signed-rank paired t-test, two-sided)
      pval_t2   : p-value from t-test 2 (classical paired t-test, two-sided)
      sig_w     : pval_w  < ALPHA
      sig_t1    : pval_t1 < ALPHA
      sig_t2    : pval_t2 < ALPHA
      d         : Cohen's d (pooled-SD formula; positive = X > Y)
      mean_x    : mean of scores_x
      mean_y    : mean of scores_y
      n         : number of paired observations
    """
    sx, sy = np.array(scores_x, dtype=float), np.array(scores_y, dtype=float)
    n = len(sx)
    if n < 3:
        return {
            "direction": 0, "pval_t2": 1.0, "pval_w": 1.0, "pval_t1": 1.0,
            "sig_t2": False, "sig_w": False, "sig_t1": False, "d": 0.0,
            "mean_x": float(np.mean(sx)) if n else 0.0,
            "mean_y": float(np.mean(sy)) if n else 0.0, "n": n,
            "is_canonical": False,
        }

    diffs = sx - sy
    mean_diff = float(np.mean(diffs))
    direction = 1 if mean_diff > 0 else (-1 if mean_diff < 0 else 0)

    if np.all(diffs == 0):
        pval_t2 = 1.0
    else:
        try:
            _, pval_t2 = stats.ttest_rel(sx, sy)
        except Exception:
            pval_t2 = 1.0

    # Wilcoxon signed-rank (primary)
    if np.all(diffs == 0):
        pval_w = 1.0
    else:
        try:
            _, pval_w = stats.wilcoxon(diffs, alternative="two-sided")
        except Exception:
            # Raises ValueError when n < 10 or all diffs are identical
            pval_w = 1.0

    nz = diffs[diffs != 0]
    if len(nz) < 2:
        pval_t1 = 1.0
    else:
        signed_ranks = np.sign(nz) * stats.rankdata(np.abs(nz))
        if np.all(signed_ranks == signed_ranks[0]):
            pval_t1 = 1.0
        else:
            try:
                _, pval_t1 = stats.ttest_1samp(signed_ranks, 0.0)
            except Exception:
                pval_t1 = 1.0
            if pval_t1 != pval_t1:   # nan guard
                pval_t1 = 1.0

    d = cohens_d(sx, sy)

    return {
        "direction":    direction,
        "pval_t2":       float(pval_t2),
        "pval_w":       float(pval_w),
        "pval_t1":   float(pval_t1),
        "sig_t2":        bool(pval_t2 < ALPHA),
        "sig_w":        bool(pval_w < ALPHA),
        "sig_t1":    bool(pval_t1 < ALPHA),
        "d":            d if d is not None else 0.0,
        "mean_x":       float(np.mean(sx)),
        "mean_y":       float(np.mean(sy)),
        "n":            n,
        "is_canonical": False,   # filled in by analyse_cell
    }


def detect_flips(per_pair, sig_key):

    flip_events   = []
    flipped_pairs = set()

    for pair, rng_results in per_pair.items():
        rng_list = list(rng_results.keys())
        for i in range(len(rng_list)):
            for j in range(i + 1, len(rng_list)):
                rA, rB = rng_list[i], rng_list[j]
                dA, dB = rng_results[rA], rng_results[rB]
                if dA[sig_key] and dB[sig_key]:
                    if dA["direction"] != 0 and dB["direction"] != 0:
                        if dA["direction"] != dB["direction"]:
                            flip_events.append((pair, rA, rB))
                            flipped_pairs.add(pair)

    return flip_events, flipped_pairs

def analyse_cell(judge, dataset, metric):
    canonical = CANONICAL[dataset][metric]

    range_data = {}
    for rng in STD_RANGES:
        sys_scores = load_system_scores(judge, dataset, metric, rng[0], rng[1])
        if sys_scores:
            range_data[rng] = sys_scores

    if not range_data:
        return {}

    all_sys_sets  = [set(rd.keys()) for rd in range_data.values()]
    common_sys    = sorted(all_sys_sets[0].intersection(*all_sys_sets[1:]))
    loaded_ranges = sorted(range_data.keys(), key=lambda r: (r[0], r[1]))

    if len(common_sys) < 2:
        return {}

    pairs = list(combinations(common_sys, 2))

    # Run all tests 
    per_pair = {}
    for (sx_id, sy_id) in pairs:
        per_pair[(sx_id, sy_id)] = {}
        for rng in loaded_ranges:
            sx_scores = range_data[rng][sx_id]
            sy_scores = range_data[rng][sy_id]
            res = compare_pair(sx_scores, sy_scores)
            res["is_canonical"] = (rng == canonical)
            per_pair[(sx_id, sy_id)][rng] = res

    delta_signed_list = []   
    delta_absmag_list = []   
    for pair in per_pair:
        dvals = [per_pair[pair][rng]["d"] for rng in loaded_ranges]
        if dvals:
            delta_signed_list.append(max(dvals) - min(dvals))
            adv = [abs(x) for x in dvals]
            delta_absmag_list.append(max(adv) - min(adv))

    # Flip detection independently per test
    _, flipped_pairs_w  = detect_flips(per_pair, "sig_w")
    _, flipped_pairs_t1 = detect_flips(per_pair, "sig_t1")
    _, flipped_pairs_t2 = detect_flips(per_pair, "sig_t2")

    return {
        "n_flip_pairs_w":     len(flipped_pairs_w),
        "n_flip_pairs_t1":    len(flipped_pairs_t1),
        "n_flip_pairs_t2":    len(flipped_pairs_t2),
        "delta_signed_list":  delta_signed_list,
        "delta_absmag_list":  delta_absmag_list,
    }


def main():
    all_results = {
        (judge, dataset, metric): analyse_cell(judge, dataset, metric)
        for dataset in ["summeval", "newsroom", "tc", "wp_a"]
        for judge   in ALL_JUDGES
        for metric  in DATASETS[dataset]["metrics"].keys()
    }

    out = os.path.join(REPORTS_DIR, "cohens_d_instability.csv")
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["judge", "dataset", "max_delta_abs_d", "mean_delta_abs_d", "n_flips"])
        for ds in ["summeval", "newsroom", "tc", "wp_a"]:
            metrics = list(DATASETS[ds]["metrics"].keys())
            for j in ALL_JUDGES:
                sig, mag, nfl = [], [], 0
                for m in metrics:
                    r = all_results.get((j, ds, m), {})
                    sig.extend(r.get("delta_signed_list", []))
                    mag.extend(r.get("delta_absmag_list", []))
                    nfl += r.get("n_flip_pairs_w", 0)
                mx = round(max(sig), 2) if sig else 0.0
                mn = round(sum(mag) / len(mag), 2) if mag else 0.0
                w.writerow([JUDGE_LABEL[j], DATASET_LABEL[ds], mx, mn, nfl])
    print(f"Saved {out}")

    tw  = sum(r.get("n_flip_pairs_w",  0) for r in all_results.values())
    tt1 = sum(r.get("n_flip_pairs_t1", 0) for r in all_results.values())
    tt2 = sum(r.get("n_flip_pairs_t2", 0) for r in all_results.values())


if __name__ == "__main__":
    main()
