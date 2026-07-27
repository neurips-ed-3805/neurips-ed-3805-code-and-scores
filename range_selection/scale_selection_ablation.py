import json, os, re, time
import numpy as np
from itertools import combinations
from collections import defaultdict

ROOT      = os.path.dirname(os.path.abspath(__file__))
SCORES_BASE  = os.path.join(os.path.dirname(ROOT), "scores")
PAIRWISE_BASE = os.path.join(os.path.dirname(ROOT), "scores", "pairwise")
OUT_DIR   = os.path.join(os.path.dirname(ROOT), "reports", "scale_selection")
os.makedirs(OUT_DIR, exist_ok=True)

JUDGES = ["gemini3_flash", "gemini31_pro",
          "gemma2_9b", "llama31_8b", "qwen3_4b", "mistral_7b",
          "gemma3_27b", "llama31_70b"]
DATASETS = ["summeval", "tc", "newsroom", "wp_a"]
CALIB_FRACS = [0.05, 0.10, 0.20, 0.50]
N_BOOTSTRAP = 1000
SEED = 42

CANONICAL_RANGES = [(1.0, 5.0), (0.0, 5.0), (1.0, 3.0), (0.0, 10.0), (0.0, 1.0)]

TC_MODEL_MAP = {
    "argmax":         "Argmax Decoding",
    "nucleus_0.3":    "Nucleus Decoding (p = 0.3)",
    "nucleus_0.5":    "Nucleus Decoding (p = 0.5)",
    "nucleus_0.7":    "Nucleus Decoding (p = 0.7)",
    "human_generated": "New Human Generated",
    "groundtruth":    "Original Ground Truth",
}

def parse_score(all_responses):
    for r in all_responses:
        try:
            v = float(str(r).strip())
            if np.isfinite(v):
                return v
        except Exception:
            pass
    return np.nan


def load_range_files(score_dir, judge):
    files = os.listdir(score_dir)
    result = defaultdict(list)
    for fn in files:
        if not fn.endswith(".json"):
            continue
        m = re.match(r"range_(.+?)(?:_run\d+)?\.json$", fn)
        if not m:
            continue
        rk = m.group(1)
        with open(os.path.join(score_dir, fn)) as f:
            result[rk].append(json.load(f))
    return dict(result)

def build_summeval(judge, metric, score_dir, pw_dir):
    range_data = load_range_files(score_dir, judge)
    range_keys  = sorted(range_data.keys())
    range_specs = []
    for rk in range_keys:
        parts = rk.split("_")
        try:
            rmin, rmax = float(parts[0]), float(parts[1])
            range_specs.append((rmin, rmax))
        except Exception:
            continue

    scores_lookup = {}   
    for ri, rk in enumerate(range_keys):
        try:
            float(rk.split("_")[0])  # skip bad keys
        except Exception:
            continue
        runs = range_data[rk]
        # average over runs
        run_scores = defaultdict(list)
        for run in runs:
            for entry in run:
                key = (entry["doc_id"], entry["system_id"])
                v = parse_score(entry.get("all_responses", []))
                run_scores[key].append(v)
        for key, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if key not in scores_lookup:
                scores_lookup[key] = np.full(len(range_keys), np.nan)
            scores_lookup[key][ri] = np.mean(valid) if valid else np.nan

    # Get all instances (doc_ids) from pairwise
    pair_dirs = os.listdir(pw_dir)
    pair_names = []
    for pd in sorted(pair_dirs):
        m = re.match(r"(M\d+)_vs_(M\d+)$", pd)
        if m:
            pair_names.append((m.group(1), m.group(2)))

    # Collect pairwise winners per pair
    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        pd = f"{ma}_vs_{mb}"
        fp = os.path.join(pw_dir, pd, "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"summeval_(.+)_{ma}_vs_{mb}$", entry["instance_id"])
            if m2:
                doc_id = m2.group(1)
                doc_winners[doc_id] = entry.get("winner_model")
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)
    NR = len(range_specs)
    NP = len(pair_names)

    inst_idx = {d: i for i, d in enumerate(instances)}

    # Build agreement matrices
    agree_matrix = np.zeros((N, NR, NP), dtype=bool)
    valid_matrix  = np.zeros((N, NR, NP), dtype=bool)

    for pi, (ma, mb) in enumerate(pair_names):
        pw = pw_raw.get((ma, mb), {})
        for n, doc in enumerate(instances):
            winner = pw.get(doc)
            if winner is None:
                continue
            pw_dir_sign = +1 if winner == ma else -1

            sa_all = scores_lookup.get((doc, ma), np.full(NR, np.nan))
            sb_all = scores_lookup.get((doc, mb), np.full(NR, np.nan))

            for ri in range(NR):
                sa, sb = sa_all[ri], sb_all[ri]
                if np.isnan(sa) or np.isnan(sb) or sa == sb:
                    continue
                valid_matrix[n, ri, pi] = True
                score_sign = +1 if sa > sb else -1
                agree_matrix[n, ri, pi] = (score_sign == pw_dir_sign)

    return instances, pair_names, agree_matrix, valid_matrix, range_specs


def build_tc(judge, metric, score_dir, pw_dir):

    range_data  = load_range_files(score_dir, judge)
    range_keys  = sorted(range_data.keys())
    range_specs = []
    for rk in range_keys:
        parts = rk.split("_")
        try:
            rmin, rmax = float(parts[0]), float(parts[1])
            range_specs.append((rmin, rmax))
        except Exception:
            continue

    scores_lookup = {}  
    NR = len(range_keys)
    for ri, rk in enumerate(range_keys):
        try:
            float(rk.split("_")[0])
        except Exception:
            continue
        runs = range_data[rk]
        run_scores = defaultdict(list)
        for run in runs:
            for doc_idx, entry in enumerate(run):
                for resp in entry.get("responses", []):
                    model_name = resp.get("model", "")
                    
                    model_key = None
                    for k, v in TC_MODEL_MAP.items():
                        if v in model_name or model_name in v:
                            model_key = k
                            break
                    if model_key is None:
                        continue
                    key = (doc_idx, model_key)
                    v = parse_score(resp.get("all_responses", []))
                    run_scores[key].append(v)
        for key, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if key not in scores_lookup:
                scores_lookup[key] = np.full(NR, np.nan)
            scores_lookup[key][ri] = np.mean(valid) if valid else np.nan

    # Parse pairwise
    pair_dirs = os.listdir(pw_dir)
    pair_names = []
    for pd in sorted(pair_dirs):
        parts = pd.split("_vs_")
        if len(parts) == 2:
            pair_names.append((parts[0], parts[1]))

    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        pd = f"{ma}_vs_{mb}"
        fp = os.path.join(pw_dir, pd, "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"tc_(\d+)_{re.escape(ma)}_vs_{re.escape(mb)}$", entry["instance_id"])
            if m2:
                doc_idx = int(m2.group(1))
                doc_winners[doc_idx] = entry.get("winner_model")
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)
    NP = len(pair_names)

    agree_matrix = np.zeros((N, NR, NP), dtype=bool)
    valid_matrix  = np.zeros((N, NR, NP), dtype=bool)

    for pi, (ma, mb) in enumerate(pair_names):
        pw = pw_raw.get((ma, mb), {})
        for n, doc_idx in enumerate(instances):
            winner = pw.get(doc_idx)
            if winner is None:
                continue
            pw_sign = +1 if winner == ma else -1
            sa_all = scores_lookup.get((doc_idx, ma), np.full(NR, np.nan))
            sb_all = scores_lookup.get((doc_idx, mb), np.full(NR, np.nan))
            for ri in range(NR):
                sa, sb = sa_all[ri], sb_all[ri]
                if np.isnan(sa) or np.isnan(sb) or sa == sb:
                    continue
                valid_matrix[n, ri, pi] = True
                agree_matrix[n, ri, pi] = ((+1 if sa > sb else -1) == pw_sign)

    return instances, pair_names, agree_matrix, valid_matrix, range_specs


def build_newsroom(judge, metric, score_dir, pw_dir):

    range_data  = load_range_files(score_dir, judge)
    range_keys  = sorted(range_data.keys())
    range_specs = []
    for rk in range_keys:
        parts = rk.split("_")
        try:
            rmin, rmax = float(parts[0]), float(parts[1])
            range_specs.append((rmin, rmax))
        except Exception:
            continue

    NR = len(range_keys)

    scores_lookup = {}
    for ri, rk in enumerate(range_keys):
        try:
            float(rk.split("_")[0])
        except Exception:
            continue
        runs = range_data[rk]
        run_scores = defaultdict(list)
        for run in runs:
            # group by doc_id preserving order
            by_doc = defaultdict(list)
            for entry in run:
                by_doc[entry["doc_id"]].append(entry)
            for doc_id, entries in by_doc.items():
                for sys_idx, entry in enumerate(entries):
                    key = (doc_id, sys_idx)
                    v = parse_score(entry.get("all_responses", []))
                    run_scores[key].append(v)
        for key, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if key not in scores_lookup:
                scores_lookup[key] = np.full(NR, np.nan)
            scores_lookup[key][ri] = np.mean(valid) if valid else np.nan

    # Parse pairwise
    pair_dirs = os.listdir(pw_dir)
    pair_names = []
    for pd in sorted(pair_dirs):
        m = re.match(r"(sys\d+)_vs_(sys\d+)$", pd)
        if m:
            pair_names.append((m.group(1), m.group(2)))

    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        pd = f"{ma}_vs_{mb}"
        fp = os.path.join(pw_dir, pd, "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"newsroom_(.+)_{re.escape(ma)}_vs_{re.escape(mb)}$", entry["instance_id"])
            if m2:
                doc_id = m2.group(1)
                doc_winners[doc_id] = entry.get("winner_model")
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)
    NP = len(pair_names)

    def sys_idx(name):
        return int(name.replace("sys", ""))

    agree_matrix = np.zeros((N, NR, NP), dtype=bool)
    valid_matrix  = np.zeros((N, NR, NP), dtype=bool)

    for pi, (ma, mb) in enumerate(pair_names):
        pw = pw_raw.get((ma, mb), {})
        ia, ib = sys_idx(ma), sys_idx(mb)
        for n, doc_id in enumerate(instances):
            winner = pw.get(doc_id)
            if winner is None:
                continue
            pw_sign = +1 if winner == ma else -1
            sa_all = scores_lookup.get((doc_id, ia), np.full(NR, np.nan))
            sb_all = scores_lookup.get((doc_id, ib), np.full(NR, np.nan))
            for ri in range(NR):
                sa, sb = sa_all[ri], sb_all[ri]
                if np.isnan(sa) or np.isnan(sb) or sa == sb:
                    continue
                valid_matrix[n, ri, pi] = True
                agree_matrix[n, ri, pi] = ((+1 if sa > sb else -1) == pw_sign)

    return instances, pair_names, agree_matrix, valid_matrix, range_specs


def build_wpa(judge, metric, score_dir, pw_dir):

    range_data  = load_range_files(score_dir, judge)
    range_keys  = sorted(range_data.keys())
    range_specs = []
    for rk in range_keys:
        parts = rk.split("_")
        try:
            rmin, rmax = float(parts[0]), float(parts[1])
            range_specs.append((rmin, rmax))
        except Exception:
            continue

    NR = len(range_keys)

    scores_lookup = {}
    for ri, rk in enumerate(range_keys):
        try:
            float(rk.split("_")[0])
        except Exception:
            continue
        runs = range_data[rk]
        run_scores = defaultdict(list)
        for run in runs:
            for idx, entry in enumerate(run):
                v = parse_score(entry.get("all_responses", []))
                run_scores[idx].append(v)
        for idx, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if idx not in scores_lookup:
                scores_lookup[idx] = np.full(NR, np.nan)
            scores_lookup[idx][ri] = np.mean(valid) if valid else np.nan

    pw_fp = os.path.join(pw_dir, "human_vs_ai", "pairwise_results.json")
    if not os.path.exists(pw_fp):
        return None
    with open(pw_fp) as f:
        pw_data = json.load(f)

    instances = []
    pw_winners = {}
    for entry in pw_data:
        m = re.match(r"wpa_prompt(\d+)_h(\d+)_a(\d+)$", entry["instance_id"])
        if not m:
            continue
        h_idx = int(m.group(2))
        a_idx = int(m.group(3))
        key = (h_idx, a_idx)
        instances.append(key)
        pw_winners[key] = entry.get("winner_model")

    N = len(instances)
    pair_names = [("human", "ai")]
    NP = 1

    agree_matrix = np.zeros((N, NR, NP), dtype=bool)
    valid_matrix  = np.zeros((N, NR, NP), dtype=bool)

    for n, (h_idx, a_idx) in enumerate(instances):
        winner = pw_winners.get((h_idx, a_idx))
        if winner is None:
            continue
        pw_sign = +1 if winner == "human" else -1
        sa_all = scores_lookup.get(h_idx, np.full(NR, np.nan))
        sb_all = scores_lookup.get(a_idx, np.full(NR, np.nan))
        for ri in range(NR):
            sa, sb = sa_all[ri], sb_all[ri]
            if np.isnan(sa) or np.isnan(sb) or sa == sb:
                continue
            valid_matrix[n, ri, 0] = True
            agree_matrix[n, ri, 0] = ((+1 if sa > sb else -1) == pw_sign)

    return instances, pair_names, agree_matrix, valid_matrix, range_specs


DATASET_BUILDER = {
    "summeval": build_summeval,
    "tc":       build_tc,
    "newsroom": build_newsroom,
    "wp_a":     build_wpa,
}

DATASET_METRICS = {
    "summeval": ["coh", "con", "flu", "rel"],
    "tc":       ["coh", "eng", "gro", "nat"],
    "newsroom": ["coh", "flu", "inf", "rel"],
    "wp_a":     ["coh", "enj", "gra", "rel"],
}

def agreement_on_subset(mask, agree_matrix, valid_matrix):
    sub_a = agree_matrix[mask]
    sub_v = valid_matrix[mask]
    n_a = sub_a.sum(axis=(0, 2))
    n_v = sub_v.sum(axis=(0, 2))
    return np.where(n_v > 0, 100.0 * n_a / n_v, np.nan)

def run_scale_selection(agree_matrix, valid_matrix, range_specs, calib_frac, n_bootstrap, seed):

    N = agree_matrix.shape[0]
    NR = len(range_specs)
    calib_size = max(1, int(round(calib_frac * N)))
    test_size  = N - calib_size

    all_mask = np.ones(N, dtype=bool)
    full_agr = agreement_on_subset(all_mask, agree_matrix, valid_matrix)

    full_agr_clean = np.where(np.isnan(full_agr), 0.0, full_agr)

    # Best canonical range
    best_canon_idx = None
    best_canon_agr = -1.0
    for cr in CANONICAL_RANGES:
        if cr in range_specs:
            ri = range_specs.index(cr)
            v = full_agr_clean[ri]
            if v > best_canon_agr:
                best_canon_agr = v
                best_canon_idx = ri

    rng = np.random.RandomState(seed)
    sel_agr_list = []
    orc_agr_list = []
    wst_agr_list = []
    sel_rank_list = []
    canon_agr_test_list = []   # canonical range agreement on test set
    selection_count = np.zeros(NR, dtype=int)

    for _ in range(n_bootstrap):
        perm = rng.permutation(N)
        calib_mask = np.zeros(N, dtype=bool)
        calib_mask[perm[:calib_size]] = True
        test_mask = ~calib_mask

        calib_agr = agreement_on_subset(calib_mask, agree_matrix, valid_matrix)
        calib_agr_c = np.where(np.isnan(calib_agr), -1.0, calib_agr)
        best_idx = np.argmax(calib_agr_c)
        selection_count[best_idx] += 1

        if test_size == 0:
            test_agr_c = full_agr_clean
        else:
            test_agr = agreement_on_subset(test_mask, agree_matrix, valid_matrix)
            test_agr_c = np.where(np.isnan(test_agr), 0.0, test_agr)

        sel_agr_list.append(test_agr_c[best_idx])
        orc_agr_list.append(test_agr_c.max())
        wst_agr_list.append(test_agr_c.min())
        rank = int((test_agr_c > test_agr_c[best_idx]).sum()) + 1
        sel_rank_list.append(rank)

        if best_canon_idx is not None:
            canon_agr_test_list.append(test_agr_c[best_canon_idx])
        else:
            canon_agr_test_list.append(0.0)

    sel_arr   = np.array(sel_agr_list)
    orc_arr   = np.array(orc_agr_list)
    wst_arr   = np.array(wst_agr_list)
    rnk_arr   = np.array(sel_rank_list)
    canon_arr = np.array(canon_agr_test_list)
    regret    = orc_arr - sel_arr
    
    canon_regret_arr = orc_arr - canon_arr
    
    vs_canon_arr = sel_arr - canon_arr

    top_ri = int(np.argmax(selection_count))

    vs_canonical = float(vs_canon_arr.mean())
    canonical_regret = float(canon_regret_arr.mean())

    return {
        "calib_size": calib_size,
        "test_size": test_size,
        "N": N,
        "n_ranges": NR,
        "sel_mean": float(sel_arr.mean()),
        "sel_std":  float(sel_arr.std()),
        "orc_mean": float(orc_arr.mean()),
        "wst_mean": float(wst_arr.mean()),
        "regret_mean": float(regret.mean()),
        "regret_median": float(np.median(regret)),
        "regret_max": float(regret.max()),
        "regret_zero_pct": float(100.0 * (regret == 0).sum() / n_bootstrap),
        "rank_mean": float(rnk_arr.mean()),
        "rank_1_pct": float(100.0 * (rnk_arr == 1).sum() / n_bootstrap),
        "top_range": range_specs[top_ri],
        "top_range_freq": float(100.0 * selection_count[top_ri] / n_bootstrap),
        "full_agr_best": float(full_agr_clean.max()),
        "full_agr_best_range": range_specs[int(np.argmax(full_agr_clean))],
        "full_agr_worst": float(full_agr_clean.min()),
        "full_agr_spread": float(full_agr_clean.max() - full_agr_clean.min()),
        "full_agr_per_range": {f"{rs[0]}_{rs[1]}": float(v) for rs, v in zip(range_specs, full_agr_clean)},
        "best_canonical_range": CANONICAL_RANGES[CANONICAL_RANGES.index(range_specs[best_canon_idx])] if best_canon_idx is not None else None,
        "best_canonical_agr_full": float(best_canon_agr),          # on full dataset
        "best_canonical_agr_test": float(canon_arr.mean()),        # on test sets (bootstrap mean)
        "canonical_regret_mean": float(canonical_regret),          # oracle - canonical (test)
        "vs_canonical_mean": float(vs_canonical),                  # selected - canonical (test)
        "vs_canonical_std": float(vs_canon_arr.std()),
    }

def main():
    t0 = time.time()
    all_results = {}

    for judge in JUDGES:
        all_results[judge] = {}
        for dataset in DATASETS:
            all_results[judge][dataset] = {}
            metrics = DATASET_METRICS[dataset]
            builder = DATASET_BUILDER[dataset]
            print(f"\n{'='*60}")
            print(f"  {judge} / {dataset}")
            print(f"{'='*60}")

            for metric in metrics:
                score_dir = os.path.join(SCORES_BASE, judge, dataset, metric)
                pw_dir    = os.path.join(PAIRWISE_BASE, judge, dataset, metric)

                if not os.path.exists(score_dir):
                    print(f"  [{metric}] MISSING score dir: {score_dir}")
                    continue
                if not os.path.exists(pw_dir):
                    print(f"  [{metric}] MISSING pairwise dir: {pw_dir}")
                    continue

                print(f"\n  Metric: {metric}")
                t1 = time.time()
                result = builder(judge, metric, score_dir, pw_dir)
                if result is None:
                    print(f"    No data.")
                    continue

                instances, pair_names, agree_matrix, valid_matrix, range_specs = result
                N  = len(instances)
                NR = len(range_specs)
                NP = len(pair_names)
                print(f"    N={N} instances, {NP} pairs, {NR} ranges")

                metric_results = {"N": N, "n_pairs": NP, "n_ranges": NR,
                                  "range_specs": range_specs, "ablation": {}}

                for frac in CALIB_FRACS:
                    calib_n = max(1, int(round(frac * N)))
                    print(f"    calib={frac:.0%} (n={calib_n}) ...", end=" ", flush=True)
                    res = run_scale_selection(
                        agree_matrix, valid_matrix, range_specs,
                        calib_frac=frac, n_bootstrap=N_BOOTSTRAP, seed=SEED
                    )
                    metric_results["ablation"][str(frac)] = res
                    print(f"regret={res['regret_mean']:.2f}pp, rank={res['rank_mean']:.1f}, "
                          f"sel={res['sel_mean']:.1f}%, vs_canon={res['vs_canonical_mean']:+.2f}pp")

                all_results[judge][dataset][metric] = metric_results
                print(f"    Done in {time.time()-t1:.1f}s")

    out_fp = os.path.join(OUT_DIR, "scale_selection_ablation_results.json")

    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [fix(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    with open(out_fp, "w") as f:
        json.dump(fix(all_results), f, indent=2)
    print(f"\nSaved: {out_fp}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
