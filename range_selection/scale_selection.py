import json, os, re, time
import numpy as np
from collections import defaultdict

ROOT          = os.path.dirname(os.path.abspath(__file__))
REPO          = os.path.dirname(ROOT)                         
SCORES_BASE   = os.path.join(REPO, "scores")
PAIRWISE_BASE = os.path.join(REPO, "scores", "pairwise")
OUT_DIR       = os.path.join(REPO, "reports", "scale_selection")
CORR_JSON     = os.path.join(REPO, "reports", "all_correlations.json")
os.makedirs(OUT_DIR, exist_ok=True)

JUDGES       = ["gemini3_flash", "gemini31_pro",
                "gemma2_9b", "llama31_8b", "qwen3_4b", "mistral_7b",
                "gemma3_27b", "llama31_70b"]
DATASETS     = ["summeval", "tc", "newsroom", "wp_a"]
CALIB_FRACS  = [0.05, 0.10, 0.20, 0.50]
N_BOOTSTRAP  = 1000
SEED         = 42

CANONICAL_RANGE: dict[tuple[str, str], tuple[float, float]] = {
    ("summeval", "coh"): (1.0, 5.0),
    ("summeval", "con"): (1.0, 5.0),
    ("summeval", "flu"): (1.0, 5.0),
    ("summeval", "rel"): (1.0, 5.0),
    ("tc", "coh"):        (1.0, 3.0),
    ("tc", "eng"):        (1.0, 3.0),
    ("tc", "gro"):        (0.0, 1.0),
    ("tc", "nat"):        (1.0, 3.0),
    ("newsroom", "coh"):  (1.0, 5.0),
    ("newsroom", "flu"):  (1.0, 5.0),
    ("newsroom", "inf"):  (1.0, 5.0),
    ("newsroom", "rel"):  (1.0, 5.0),
    ("wp_a", "coh"):      (1.0, 5.0),
    ("wp_a", "enj"):      (1.0, 5.0),
    ("wp_a", "gra"):      (1.0, 5.0),
    ("wp_a", "rel"):      (1.0, 5.0),
}

STANDARD_RANGES: list[tuple[float, float]] = [
    (0.0, 1.0),
    (1.0, 3.0),
    (1.0, 5.0),
    (0.0, 10.0),
    (0.0, 100.0),
]

TC_MODEL_MAP = {
    "argmax":          "Argmax Decoding",
    "nucleus_0.3":     "Nucleus Decoding (p = 0.3)",
    "nucleus_0.5":     "Nucleus Decoding (p = 0.5)",
    "nucleus_0.7":     "Nucleus Decoding (p = 0.7)",
    "human_generated": "New Human Generated",
    "groundtruth":     "Original Ground Truth",
}

DATASET_METRICS = {
    "summeval": ["coh", "con", "flu", "rel"],
    "tc":       ["coh", "eng", "gro", "nat"],
    "newsroom": ["coh", "flu", "inf", "rel"],
    "wp_a":     ["coh", "enj", "gra", "rel"],
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


def load_range_files(score_dir):
    result = defaultdict(list)
    for fn in os.listdir(score_dir):
        if not fn.endswith(".json"):
            continue
        m = re.match(r"range_(.+?)(?:_run\d+)?\.json$", fn)
        if not m:
            continue
        with open(os.path.join(score_dir, fn)) as f:
            result[m.group(1)].append(json.load(f))
    return dict(result)


def parse_range_spec(range_key):
    m = re.match(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", range_key)
    if not m:
        return None
    try:
        return (float(m.group(1)), float(m.group(2)))
    except Exception:
        return None


def build_summeval(judge, metric, score_dir, pw_dir):
    range_data = load_range_files(score_dir)
    range_specs = []
    rk_to_ri = {}
    for rk in sorted(range_data.keys()):
        rs = parse_range_spec(rk)
        if rs is None:
            continue
        rk_to_ri[rk] = len(range_specs)
        range_specs.append(rs)
    NR = len(range_specs)
    if NR == 0:
        return None

    systems = set()
    pair_names = []
    for pd in sorted(os.listdir(pw_dir)):
        m = re.match(r"(M\d+)_vs_(M\d+)$", pd)
        if m:
            pair_names.append((m.group(1), m.group(2)))
            systems.update([m.group(1), m.group(2)])
    if not systems:
        return None
    systems = sorted(systems, key=lambda x: int(x[1:]))
    K = len(systems)
    sys_idx = {s: i for i, s in enumerate(systems)}

    scores_lookup = {}
    for rk, runs in range_data.items():
        ri = rk_to_ri.get(rk)
        if ri is None:
            continue
        run_scores = defaultdict(list)
        for run in runs:
            for entry in run:
                key = (entry["doc_id"], entry["system_id"])
                v = parse_score(entry.get("all_responses", []))
                run_scores[key].append(v)
        for key, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if key not in scores_lookup:
                scores_lookup[key] = np.full(NR, np.nan)
            scores_lookup[key][ri] = np.mean(valid) if valid else np.nan

    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        fp = os.path.join(pw_dir, f"{ma}_vs_{mb}", "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"summeval_(.+)_{re.escape(ma)}_vs_{re.escape(mb)}$",
                          entry["instance_id"])
            if m2:
                doc_winners[m2.group(1)] = entry["winner_model"]
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)

    score_matrix  = np.full((N, NR, K), np.nan)
    pw_wins       = np.zeros((N, K))
    pw_comps      = np.zeros((N, K))
    pw_pair_wins  = np.zeros((N, K, K), dtype=np.float32)

    for n, doc_id in enumerate(instances):
        for ki, sys in enumerate(systems):
            arr = scores_lookup.get((doc_id, sys))
            if arr is not None:
                score_matrix[n, :, ki] = arr

        for ma, mb in pair_names:
            pw = pw_raw.get((ma, mb), {})
            winner = pw.get(doc_id)
            if winner is None:
                continue
            ia, ib = sys_idx[ma], sys_idx[mb]
            pw_comps[n, ia] += 1
            pw_comps[n, ib] += 1
            if winner == ma:
                pw_wins[n, ia] += 1
                pw_pair_wins[n, ia, ib] += 1
            elif winner == mb:
                pw_wins[n, ib] += 1
                pw_pair_wins[n, ib, ia] += 1

    pw_pair_names = sorted(set((min(ia, ib), max(ia, ib)) for ia, ib in
                               [(sys_idx[ma], sys_idx[mb]) for ma, mb in pair_names]))

    return instances, systems, score_matrix, pw_wins, pw_comps, range_specs, pw_pair_wins, pw_pair_names


def build_tc(judge, metric, score_dir, pw_dir):
    range_data = load_range_files(score_dir)
    range_specs = []
    rk_to_ri = {}
    for rk in sorted(range_data.keys()):
        rs = parse_range_spec(rk)
        if rs is None:
            continue
        rk_to_ri[rk] = len(range_specs)
        range_specs.append(rs)
    NR = len(range_specs)
    if NR == 0:
        return None

    systems = list(TC_MODEL_MAP.keys())
    K = len(systems)
    sys_idx = {s: i for i, s in enumerate(systems)}

    scores_lookup = {}
    for rk, runs in range_data.items():
        ri = rk_to_ri.get(rk)
        if ri is None:
            continue
        run_scores = defaultdict(list)
        for run in runs:
            for doc_idx, entry in enumerate(run):
                for resp in entry.get("responses", []):
                    model_name = resp.get("model", "")
                    model_key = next(
                        (k for k, v in TC_MODEL_MAP.items()
                         if v in model_name or model_name in v), None)
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

    pair_names = []
    for pd in sorted(os.listdir(pw_dir)):
        parts = pd.split("_vs_")
        if len(parts) == 2:
            pair_names.append((parts[0], parts[1]))

    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        fp = os.path.join(pw_dir, f"{ma}_vs_{mb}", "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"tc_(\d+)_{re.escape(ma)}_vs_{re.escape(mb)}$",
                          entry["instance_id"])
            if m2:
                doc_winners[int(m2.group(1))] = entry["winner_model"]
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)

    score_matrix  = np.full((N, NR, K), np.nan)
    pw_wins       = np.zeros((N, K))
    pw_comps      = np.zeros((N, K))
    pw_pair_wins  = np.zeros((N, K, K), dtype=np.float32)

    valid_pairs = [(ma, mb) for ma, mb in pair_names if ma in sys_idx and mb in sys_idx]

    for n, doc_idx in enumerate(instances):
        for ki, sys in enumerate(systems):
            arr = scores_lookup.get((doc_idx, sys))
            if arr is not None:
                score_matrix[n, :, ki] = arr

        for ma, mb in valid_pairs:
            pw = pw_raw.get((ma, mb), {})
            winner = pw.get(doc_idx)
            if winner is None:
                continue
            ia, ib = sys_idx[ma], sys_idx[mb]
            pw_comps[n, ia] += 1
            pw_comps[n, ib] += 1
            if winner == ma:
                pw_wins[n, ia] += 1
                pw_pair_wins[n, ia, ib] += 1
            elif winner == mb:
                pw_wins[n, ib] += 1
                pw_pair_wins[n, ib, ia] += 1

    pw_pair_names = sorted(set((min(sys_idx[ma], sys_idx[mb]),
                                max(sys_idx[ma], sys_idx[mb]))
                               for ma, mb in valid_pairs))

    return instances, systems, score_matrix, pw_wins, pw_comps, range_specs, pw_pair_wins, pw_pair_names


def build_newsroom(judge, metric, score_dir, pw_dir):
    range_data = load_range_files(score_dir)
    range_specs = []
    rk_to_ri = {}
    for rk in sorted(range_data.keys()):
        rs = parse_range_spec(rk)
        if rs is None:
            continue
        rk_to_ri[rk] = len(range_specs)
        range_specs.append(rs)
    NR = len(range_specs)
    if NR == 0:
        return None

    sys_set = set()
    pair_names = []
    for pd in sorted(os.listdir(pw_dir)):
        m = re.match(r"(sys\d+)_vs_(sys\d+)$", pd)
        if m:
            pair_names.append((m.group(1), m.group(2)))
            sys_set.update([m.group(1), m.group(2)])
    systems = sorted(sys_set, key=lambda x: int(x[3:]))
    K = len(systems)
    sys_idx = {s: i for i, s in enumerate(systems)}

    scores_lookup = {}
    for rk, runs in range_data.items():
        ri = rk_to_ri.get(rk)
        if ri is None:
            continue
        run_scores = defaultdict(list)
        for run in runs:
            by_doc = defaultdict(list)
            for entry in run:
                by_doc[entry["doc_id"]].append(entry)
            for doc_id, entries in by_doc.items():
                for pos, entry in enumerate(entries):
                    key = (doc_id, pos)
                    v = parse_score(entry.get("all_responses", []))
                    run_scores[key].append(v)
        for key, vs in run_scores.items():
            valid = [v for v in vs if not np.isnan(v)]
            if key not in scores_lookup:
                scores_lookup[key] = np.full(NR, np.nan)
            scores_lookup[key][ri] = np.mean(valid) if valid else np.nan

    pw_raw = {}
    all_doc_sets = []
    for ma, mb in pair_names:
        fp = os.path.join(pw_dir, f"{ma}_vs_{mb}", "pairwise_results.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        doc_winners = {}
        for entry in data:
            m2 = re.match(rf"newsroom_(.+)_{re.escape(ma)}_vs_{re.escape(mb)}$",
                          entry["instance_id"])
            if m2:
                doc_winners[m2.group(1)] = entry["winner_model"]
        pw_raw[(ma, mb)] = doc_winners
        all_doc_sets.append(set(doc_winners.keys()))

    if not all_doc_sets:
        return None
    instances = sorted(set.intersection(*all_doc_sets))
    N = len(instances)

    score_matrix  = np.full((N, NR, K), np.nan)
    pw_wins       = np.zeros((N, K))
    pw_comps      = np.zeros((N, K))
    pw_pair_wins  = np.zeros((N, K, K), dtype=np.float32)

    for n, doc_id in enumerate(instances):
        for ki, sys in enumerate(systems):
            pos = int(sys[3:])
            arr = scores_lookup.get((doc_id, pos))
            if arr is not None:
                score_matrix[n, :, ki] = arr

        for ma, mb in pair_names:
            pw = pw_raw.get((ma, mb), {})
            winner = pw.get(doc_id)
            if winner is None:
                continue
            ia, ib = sys_idx[ma], sys_idx[mb]
            pw_comps[n, ia] += 1
            pw_comps[n, ib] += 1
            if winner == ma:
                pw_wins[n, ia] += 1
                pw_pair_wins[n, ia, ib] += 1
            elif winner == mb:
                pw_wins[n, ib] += 1
                pw_pair_wins[n, ib, ia] += 1

    pw_pair_names = sorted(set((min(sys_idx[ma], sys_idx[mb]),
                                max(sys_idx[ma], sys_idx[mb]))
                               for ma, mb in pair_names))

    return instances, systems, score_matrix, pw_wins, pw_comps, range_specs, pw_pair_wins, pw_pair_names


def build_wpa(judge, metric, score_dir, pw_dir):
    range_data = load_range_files(score_dir)
    range_specs = []
    rk_to_ri = {}
    for rk in sorted(range_data.keys()):
        rs = parse_range_spec(rk)
        if rs is None:
            continue
        rk_to_ri[rk] = len(range_specs)
        range_specs.append(rs)
    NR = len(range_specs)
    if NR == 0:
        return None

    scores_lookup = {}
    for rk, runs in range_data.items():
        ri = rk_to_ri.get(rk)
        if ri is None:
            continue
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

    systems = ["human", "ai"]
    fp = os.path.join(pw_dir, "human_vs_ai", "pairwise_results.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        pw_data = json.load(f)

    instances = []
    pw_winners = {}
    for entry in pw_data:
        m = re.match(r"wpa_prompt(\d+)_h(\d+)_a(\d+)$", entry["instance_id"])
        if not m:
            continue
        h_idx, a_idx = int(m.group(2)), int(m.group(3))
        key = (h_idx, a_idx)
        instances.append(key)
        pw_winners[key] = entry["winner_model"]

    N = len(instances)
    score_matrix  = np.full((N, NR, 2), np.nan)
    pw_wins       = np.zeros((N, 2))
    pw_comps      = np.zeros((N, 2))
    pw_pair_wins  = np.zeros((N, 2, 2), dtype=np.float32)

    for n, (h_idx, a_idx) in enumerate(instances):
        h_arr = scores_lookup.get(h_idx)
        a_arr = scores_lookup.get(a_idx)
        if h_arr is not None:
            score_matrix[n, :, 0] = h_arr
        if a_arr is not None:
            score_matrix[n, :, 1] = a_arr

        winner = pw_winners.get((h_idx, a_idx))
        if winner is not None:
            pw_comps[n, 0] += 1
            pw_comps[n, 1] += 1
            if winner == "human":
                pw_wins[n, 0] += 1
                pw_pair_wins[n, 0, 1] += 1
            elif winner == "ai":
                pw_wins[n, 1] += 1
                pw_pair_wins[n, 1, 0] += 1

    pw_pair_names = [(0, 1)]

    return instances, systems, score_matrix, pw_wins, pw_comps, range_specs, pw_pair_wins, pw_pair_names


DATASET_BUILDER = {
    "summeval": build_summeval,
    "tc":       build_tc,
    "newsroom": build_newsroom,
    "wp_a":     build_wpa,
}


def precompute_spa3(score_matrix, pw_pair_wins, pw_pair_names, std_ris):

    N, NR, K = score_matrix.shape
    n_std = len(std_ris)
    pair_ii = np.array([p[0] for p in pw_pair_names], dtype=int)
    pair_jj = np.array([p[1] for p in pw_pair_names], dtype=int)
    P = len(pair_ii)

    pw_ia_wins = pw_pair_wins[:, pair_ii, pair_jj].astype(float)  
    pw_ib_wins = pw_pair_wins[:, pair_jj, pair_ii].astype(float)  
    pw_net = pw_ia_wins - pw_ib_wins                               

    sc_sub  = score_matrix[:, std_ris, :]                          
    sc_diff = sc_sub[:, :, pair_ii] - sc_sub[:, :, pair_jj]       

    pw_valid = (pw_net != 0)                                        
    sc_valid = (~np.isnan(sc_diff)) & (sc_diff != 0)               

    product = np.sign(pw_net[:, np.newaxis, :]) * np.sign(sc_diff)  

    both_valid = pw_valid[:, np.newaxis, :] & sc_valid              
    agr_mat = np.where(both_valid, (product > 0).astype(float), np.nan)

    n_valid = np.sum(both_valid, axis=(0, 2))                       

    return agr_mat, pair_ii, pair_jj, n_valid


def _spa3_agr(agr_slice):

    return np.nanmean(agr_slice.reshape(agr_slice.shape[0], agr_slice.shape[1], -1),
                      axis=(0, 2))


def run_spa3_selection(agr_mat, range_specs, std_ris, std_specs,
                       canon_ri, calib_frac, n_bootstrap, seed):

    N = agr_mat.shape[0]
    n_std = len(std_ris)
    calib_size = max(1, int(round(calib_frac * N)))
    test_size  = N - calib_size

    full_agr = _spa3_agr(agr_mat)                                  
    full_agr_c = np.where(np.isnan(full_agr), -1.0, full_agr)      

    oracle_local = int(np.argmax(full_agr_c))
    oracle_agr   = float(full_agr_c[oracle_local])
    oracle_spec  = std_specs[oracle_local]

    canon_local    = next((j for j, ri in enumerate(std_ris) if ri == canon_ri), None)
    canon_agr_full = float(full_agr_c[canon_local]) if canon_local is not None else float("nan")
    gain_oracle    = oracle_agr - canon_agr_full

    rng       = np.random.RandomState(seed)
    sel_list  = []
    orc_list  = []
    can_list  = []
    rank_list = []
    sel_count = np.zeros(n_std, dtype=int)

    for _ in range(n_bootstrap):
        perm      = rng.permutation(N)
        calib_idx = perm[:calib_size]
        test_idx  = perm[calib_size:]

        c_agr   = _spa3_agr(agr_mat[calib_idx])
        c_agr_c = np.where(np.isnan(c_agr), -1.0, c_agr)
        best_local = int(np.argmax(c_agr_c))
        sel_count[best_local] += 1

        if test_size == 0:
            eval_agr_c = full_agr_c
        else:
            t_agr   = _spa3_agr(agr_mat[test_idx])
            eval_agr_c = np.where(np.isnan(t_agr), -1.0, t_agr)

        sel_list.append(float(eval_agr_c[best_local]))
        orc_list.append(float(eval_agr_c.max()))
        rank_list.append(int((eval_agr_c > eval_agr_c[best_local]).sum()) + 1)
        can_list.append(float(eval_agr_c[canon_local]) if canon_local is not None
                        else float(eval_agr_c.max()))

    sel_arr  = np.array(sel_list)
    orc_arr  = np.array(orc_list)
    can_arr  = np.array(can_list)
    rank_arr = np.array(rank_list)
    regret   = orc_arr - sel_arr
    vs_canon = sel_arr - can_arr
    top_local = int(np.argmax(sel_count))

    mv_agr_full = float(full_agr_c[top_local])
    mv_vs_canon = mv_agr_full - canon_agr_full if not np.isnan(canon_agr_full) else float("nan")
    mv_is_oracle = (top_local == oracle_local)

    return {
        "N": N, "calib_size": calib_size, "test_size": test_size,
        "n_standard_ranges": n_std,

        "oracle_std_range":            list(oracle_spec),
        "oracle_std_agr":              float(oracle_agr),
        "canonical_agr_full":          float(canon_agr_full),
        "gain_oracle_over_canonical":  float(gain_oracle),
        "all_std_agr": {
            f"{rs[0]}_{rs[1]}": float(full_agr_c[j])
            for j, rs in enumerate(std_specs)
        },

        "mv_range":      list(std_specs[top_local]),
        "mv_freq":       float(100.0 * sel_count[top_local] / n_bootstrap),
        "mv_agr_full":   float(mv_agr_full),
        "mv_vs_canon":   float(mv_vs_canon),
        "mv_is_oracle":  bool(mv_is_oracle),

        "sel_mean":          float(sel_arr.mean()),
        "sel_std":           float(sel_arr.std()),
        "orc_mean":          float(orc_arr.mean()),
        "regret_mean":       float(regret.mean()),
        "regret_median":     float(np.median(regret)),
        "regret_max":        float(regret.max()),
        "regret_zero_pct":   float(100.0 * (regret <= 0).sum() / n_bootstrap),
        "rank_mean":         float(rank_arr.mean()),
        "rank_1_pct":        float(100.0 * (rank_arr == 1).sum() / n_bootstrap),
        "vs_canonical_mean": float(vs_canon.mean()),
        "vs_canonical_std":  float(vs_canon.std()),
        "top_selected_range": list(std_specs[top_local]),
        "top_selected_freq":  float(100.0 * sel_count[top_local] / n_bootstrap),
        "selection_dist": {
            f"{rs[0]}_{rs[1]}": float(100.0 * sel_count[j] / n_bootstrap)
            for j, rs in enumerate(std_specs)
        },
    }

def load_human_corr():

    if not os.path.exists(CORR_JSON):
        print(f"  [warn] CORR_JSON not found: {CORR_JSON}")
        return {}
    with open(CORR_JSON) as f:
        raw = json.load(f)
    out = {}
    for key, val in raw.items():
        parts = key.split("|")
        if len(parts) != 5:
            continue
        judge, ds, met, rlo, rhi = parts
        try:
            out[(judge, ds, met, float(rlo), float(rhi))] = val
        except Exception:
            pass
    return out


def human_oracle_for(judge, dataset, metric, hcorr):

    ktau_per = {}
    for rs in STANDARD_RANGES:
        k = (judge, dataset, metric, rs[0], rs[1])
        if k in hcorr:
            ktau_per[rs] = hcorr[k].get("kendalltau", float("nan"))
    if not ktau_per:
        return None
    best_rng = max(ktau_per, key=lambda r: ktau_per[r])
    canon_spec = CANONICAL_RANGE.get((dataset, metric))
    canon_ktau = ktau_per.get(canon_spec, float("nan"))
    return best_rng, ktau_per[best_rng], canon_ktau, ktau_per


def fix_json(obj):
    """Recursively convert numpy scalars to native types for json.dump."""
    if isinstance(obj, dict):
        return {k: fix_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [fix_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj


def main():
    t0 = time.time()
    all_results = {}
    hcorr = load_human_corr()
    print(f"Loaded human correlation entries: {len(hcorr)}")

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
                    print(f"  [{metric}] MISSING score dir"); continue
                if not os.path.exists(pw_dir):
                    print(f"  [{metric}] MISSING pairwise dir"); continue

                print(f"\n  Metric: {metric}")
                t1 = time.time()
                result = builder(judge, metric, score_dir, pw_dir)
                if result is None:
                    print(f"    No data."); continue

                (instances, systems, score_matrix, pw_wins, pw_comps,
                 range_specs, pw_pair_wins, pw_pair_names) = result
                N, NR, K = score_matrix.shape
                P = len(pw_pair_names)
                print(f"    N={N} docs, K={K} systems, {NR} ranges, {P} pairs")

                # Canonical range index
                canon_spec = CANONICAL_RANGE.get((dataset, metric))
                canon_ri = next(
                    (i for i, rs in enumerate(range_specs) if rs == canon_spec), None
                )
                print(f"    canonical={canon_spec}, ri={canon_ri}")

                # Standard range indices
                std_set   = set(STANDARD_RANGES)
                std_ris   = [i for i, rs in enumerate(range_specs) if rs in std_set]
                std_specs = [range_specs[i] for i in std_ris]
                if not std_ris:
                    print(f"    No standard ranges found."); continue

                # Precompute agreement matrix once
                print(f"    Precomputing SPA3 agr_mat...", end=" ", flush=True)
                agr_mat, pair_ii, pair_jj, n_valid = precompute_spa3(
                    score_matrix, pw_pair_wins, pw_pair_names, std_ris
                )
                print(f"done ({time.time()-t1:.1f}s)  shape agr={agr_mat.shape}  "
                      f"valid_instances={n_valid.tolist()}")

                # Human correlation oracle
                hinfo = human_oracle_for(judge, dataset, metric, hcorr)

                metric_results = {
                    "N": N, "K": K, "P": P,
                    "systems": systems,
                    "n_ranges": NR, "range_specs": range_specs,
                    "canonical_spec": list(canon_spec) if canon_spec else None,
                    "human_oracle": None,
                    "spa3": {},
                }

                if hinfo is not None:
                    best_rng, best_ktau, canon_ktau, ktau_per = hinfo
                    metric_results["human_oracle"] = {
                        "best_range":     list(best_rng),
                        "best_ktau":      float(best_ktau),
                        "canonical_ktau": float(canon_ktau),
                        "ktau_per_range": {f"{r[0]}_{r[1]}": float(v)
                                           for r, v in ktau_per.items()},
                    }
                    print(f"    Human oracle: [{best_rng[0]:.4g},{best_rng[1]:.4g}]"
                          f" ktau={best_ktau:.3f}  canonical ktau={canon_ktau:.3f}")

                for frac in CALIB_FRACS:
                    calib_n = max(1, int(round(frac * N)))
                    t2 = time.time()
                    res = run_spa3_selection(
                        agr_mat, range_specs, std_ris, std_specs,
                        canon_ri, calib_frac=frac, n_bootstrap=N_BOOTSTRAP, seed=SEED
                    )
                    metric_results["spa3"][str(frac)] = res

                    orc = res["oracle_std_range"]
                    mv  = res["mv_range"]
                    print(f"    calib={frac:.0%}(n={calib_n}): "
                          f"oracle=[{orc[0]:.4g},{orc[1]:.4g}] Agr={res['oracle_std_agr']:.3f} "
                          f"canon_Agr={res['canonical_agr_full']:.3f} "
                          f"gain={res['gain_oracle_over_canonical']:+.3f} "
                          f"MV=[{mv[0]:.4g},{mv[1]:.4g}] MV_Agr={res['mv_agr_full']:.3f} "
                          f"({time.time()-t2:.1f}s)")

                # Annotate human correlation for MV-selected ranges
                if hinfo is not None:
                    _, _, _, ktau_per = hinfo
                    for frac in CALIB_FRACS:
                        res = metric_results["spa3"].get(str(frac), {})
                        if not res:
                            continue
                        mv_t = tuple(res["mv_range"])
                        res["mv_human_ktau"]         = float(ktau_per.get(mv_t, float("nan")))
                        res["mv_beats_canon_human"]  = (
                            res["mv_human_ktau"] >
                            metric_results["human_oracle"]["canonical_ktau"] + 0.01)
                        res["mv_matches_hum_oracle"] = (mv_t == tuple(
                            metric_results["human_oracle"]["best_range"]))
                        oracle_t = tuple(res["oracle_std_range"])
                        res["oracle_human_ktau"] = float(ktau_per.get(oracle_t, float("nan")))

                all_results[judge][dataset][metric] = metric_results
                print(f"    Metric total: {time.time()-t1:.1f}s")

    out_fp = os.path.join(OUT_DIR, "scale_selection_results.json")
    with open(out_fp, "w") as f:
        json.dump(fix_json(all_results), f, indent=2)
    print(f"\nSaved JSON: {out_fp}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")



if __name__ == "__main__":
    main()
