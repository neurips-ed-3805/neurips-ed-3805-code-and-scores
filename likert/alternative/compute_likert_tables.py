import os, csv

import numpy as np

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE       = os.path.dirname(os.path.abspath(__file__))
LIKERT_CSV = os.path.join(HERE, "likert_correlations.csv")
CONT_CSV   = os.path.join(_ROOT, "reports", "standard_range_correlations.csv")

DSM = {
    "summeval": ["coh", "con", "flu", "rel"],
    "newsroom": ["coh", "flu", "inf", "rel"],
    "tc":       ["coh", "eng", "gro", "nat"],
    "wp_a":     ["coh", "enj", "gra", "rel"],
}
SCALES = [3, 5, 7]
JUDGES = [
    "gemma2_9b", "gemma3_27b", "qwen3_4b", "mistral_7b", "llama31_8b", "llama31_70b",
    "gemini2_flash", "gemini3_flash", "gemini31_pro",
]


def load_likert():
    if not os.path.exists(LIKERT_CSV):
        raise SystemExit(f"{LIKERT_CSV} not found -- run likert/alternative/compute_likert_correlations.py first.")
    d = {}
    for r in csv.DictReader(open(LIKERT_CSV)):
        d[(r["judge"], r["dataset"], r["metric"], int(r["scale"]))] = (float(r["spearman"]), float(r["kendall"]))
    return d
LIKERT = load_likert()


def load_cont():
    d = {}
    for r in csv.DictReader(open(CONT_CSV)):
        d.setdefault((r["model"], r["dataset"], r["metric"]), []).append({
            "sp": float(r["spearman"]), "kt": float(r["kendalltau"]),
            "canon": str(r["is_canon"]).strip().lower() == "true"})
    return d
CONT = load_cont()


def cont_minmax(judge, ds):
    mn_sp, mn_kt, mx_sp, mx_kt, base = [], [], [], [], []
    for m in DSM[ds]:
        rows = CONT.get((judge, ds, m), [])
        if not rows:
            continue
        mn_sp.append(min(x["sp"] for x in rows)); mx_sp.append(max(x["sp"] for x in rows))
        mn_kt.append(min(x["kt"] for x in rows)); mx_kt.append(max(x["kt"] for x in rows))
        can = [x["sp"] for x in rows if x["canon"]]
        base.append(can[0] if can else np.nan)
    if not mn_sp:
        return (np.nan,) * 5
    return (np.mean(mn_sp), np.mean(mn_kt), np.mean(mx_sp), np.mean(mx_kt), np.nanmean(base))


def likert_avg(judge, ds, scale):
    sp = [c[0] for m in DSM[ds] if (c := LIKERT.get((judge, ds, m, scale)))]
    kt = [c[1] for m in DSM[ds] if (c := LIKERT.get((judge, ds, m, scale)))]
    return (np.mean(sp), np.mean(kt)) if sp else (np.nan, np.nan)


def likert_peak(judge, ds):
    peaks = []
    for m in DSM[ds]:
        rs = [c[0] for s in SCALES if (c := LIKERT.get((judge, ds, m, s)))]
        if rs:
            peaks.append(max(rs))
    return np.mean(peaks) if peaks else np.nan


def f3(x):
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"


def main():
    rows2 = []
    for judge in JUDGES:
        for ds in DSM:
            if all(not CONT.get((judge, ds, m)) for m in DSM[ds]):
                continue
            lik = {s: likert_avg(judge, ds, s) for s in SCALES}
            mn_sp, mn_kt, mx_sp, mx_kt, _ = cont_minmax(judge, ds)
            rows2.append([judge, ds,
                          f3(lik[3][0]), f3(lik[3][1]), f3(lik[5][0]), f3(lik[5][1]),
                          f3(lik[7][0]), f3(lik[7][1]),
                          f3(mn_sp), f3(mn_kt), f3(mx_sp), f3(mx_kt)])
    with open(os.path.join(HERE, "likert_table_values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset",
                    "likert3_spearman", "likert3_kendall", "likert5_spearman", "likert5_kendall",
                    "likert7_spearman", "likert7_kendall",
                    "cont_min_spearman", "cont_min_kendall", "cont_max_spearman", "cont_max_kendall"])
        w.writerows(rows2)

    rows3 = []
    agg = {j: {"lik": [], "delta": []} for j in JUDGES}
    for judge in JUDGES:
        for ds in DSM:
            if all(not CONT.get((judge, ds, m)) for m in DSM[ds]):
                continue
            lik   = likert_peak(judge, ds)
            base  = cont_minmax(judge, ds)[4]
            delta = lik - base
            rows3.append([judge, ds, f3(lik), f"{delta:+.3f}"])
            agg[judge]["lik"].append(lik); agg[judge]["delta"].append(delta)
    for judge in JUDGES:
        if agg[judge]["lik"]:
            rows3.append([judge, "OVERALL", f3(np.mean(agg[judge]["lik"])),
                          f"{np.mean(agg[judge]['delta']):+.3f}"])
    with open(os.path.join(HERE, "likert_peak_values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset", "peak_likert_spearman", "delta_vs_canonical"])
        w.writerows(rows3)

    print(f"Wrote {len(rows2)} rows -> likert_table_values.csv")
    print(f"Wrote {len(rows3)} rows -> likert_peak_values.csv")


if __name__ == "__main__":
    main()
