import os
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL_DIR = os.path.join(ROOT, "reports", "scale_selection")
REPORTS = os.path.join(ROOT, "reports")
FRACS = ["0.05", "0.1", "0.2", "0.5"]


def rng(spec):
    f = lambda v: str(int(v)) if float(v) == int(float(v)) else str(v)
    return f"[{f(spec[0])},{f(spec[1])}]"


def table5():
    data = json.load(open(os.path.join(SEL_DIR, "scale_selection_results.json")))
    rows = []
    for judge in data:
        for ds in data[judge]:
            for m in data[judge][ds]:
                s = data[judge][ds][m].get("spa3", {}).get("0.2")
                if not s:
                    continue
                valid = [v for v in s["all_std_agr"].values() if v >= 0]  # exclude degenerate ranges
                spread = (max(valid) - min(valid)) * 100 if valid else float("nan")
                rows.append([judge, ds, m,
                             rng(s["oracle_std_range"]),
                             f"{s['oracle_std_agr'] * 100:.1f}",
                             f"{spread:.1f}",
                             f"{s['regret_mean'] * 100:.2f}",
                             f"{(s['oracle_std_agr'] - s['canonical_agr_full']) * 100:+.1f}"])
    out = os.path.join(REPORTS, "scale_selection.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset", "metric", "best_range", "best_agreement_pct",
                    "spread_pp", "regret_pp", "vs_canonical_pp"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


def table6():
    data = json.load(open(os.path.join(SEL_DIR, "scale_selection_ablation_results.json")))
    rows = []
    for judge in data:
        for ds in data[judge]:
            vals = []
            for fr in FRACS:
                regs = [data[judge][ds][m]["ablation"][fr]["regret_mean"]
                        for m in data[judge][ds] if fr in data[judge][ds][m].get("ablation", {})]
                vals.append(sum(regs) / len(regs) if regs else float("nan"))
            rows.append([judge, ds] + [f"{v:.2f}" for v in vals])
    out = os.path.join(REPORTS, "scale_selection_ablation.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset", "regret_5pct", "regret_10pct", "regret_20pct", "regret_50pct"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    table5()
    table6()
