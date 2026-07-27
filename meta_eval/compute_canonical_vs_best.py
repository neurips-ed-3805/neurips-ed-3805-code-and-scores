import os
import csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(ROOT, "reports", "standard_range_correlations.csv")
OUT_CSV = os.path.join(ROOT, "reports", "canonical_vs_best.csv")

DATASET_METRICS = {
    "summeval": ["coh", "con", "flu", "rel"],
    "newsroom": ["coh", "flu", "inf", "rel"],
    "tc":       ["coh", "eng", "gro", "nat"],
    "wp_a":     ["coh", "enj", "gra", "rel"],
}


def load():
    rows = defaultdict(list)   # (model,dataset,metric) -> [{range,is_canon,sp,kt}]
    with open(IN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["dataset"], r["metric"])].append({
                "range": (r["rmin"], r["rmax"]),
                "is_canon": r["is_canon"].strip().lower() == "true",
                "sp": float(r["spearman"]), "kt": float(r["kendalltau"]),
            })
    return rows


def main():
    rows = load()
    models = sorted({k[0] for k in rows})
    out = []
    for model in models:
        j_can_sp, j_can_kt, j_best_sp, j_best_kt = [], [], [], []

        for ds, metrics in DATASET_METRICS.items():
            present = [m for m in metrics if (model, ds, m) in rows]
            if not present:
                continue

            can_sp = [next(x["sp"] for x in rows[(model, ds, m)] if x["is_canon"]) for m in present]
            can_kt = [next(x["kt"] for x in rows[(model, ds, m)] if x["is_canon"]) for m in present]

            # candidate = best NON-canonical standard range (single range, metric-averaged)
            canon_ranges = {x["range"] for m in present for x in rows[(model, ds, m)] if x["is_canon"]}
            range_sp, range_kt = defaultdict(list), defaultdict(list)

            for m in present:
                for x in rows[(model, ds, m)]:
                    if x["range"] in canon_ranges:
                        continue
                    range_sp[x["range"]].append(x["sp"])
                    range_kt[x["range"]].append(x["kt"])

            best_r = max(range_sp, key=lambda r: sum(range_sp[r]) / len(range_sp[r]))
            can_sp_avg, can_kt_avg = sum(can_sp) / len(can_sp), sum(can_kt) / len(can_kt)

            best_sp_avg = sum(range_sp[best_r]) / len(range_sp[best_r])
            best_kt_avg = sum(range_kt[best_r]) / len(range_kt[best_r])
            out.append([model, ds, f"{can_sp_avg:.3f}", f"{can_kt_avg:.3f}",f"[{best_r[0]},{best_r[1]}]", f"{best_sp_avg:.3f}", f"{best_kt_avg:.3f}"])

            j_can_sp.append(can_sp_avg); j_can_kt.append(can_kt_avg)
            j_best_sp.append(best_sp_avg); j_best_kt.append(best_kt_avg)

        if j_can_sp:
            out.append([model, "AVG",
                        f"{sum(j_can_sp)/len(j_can_sp):.3f}", f"{sum(j_can_kt)/len(j_can_kt):.3f}",
                        "-",
                        f"{sum(j_best_sp)/len(j_best_sp):.3f}", f"{sum(j_best_kt)/len(j_best_kt):.3f}"])
            
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset", "canonical_spearman", "canonical_kendall",
                    "best_std_range", "best_std_spearman", "best_std_kendall"])
        w.writerows(out)
        
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
