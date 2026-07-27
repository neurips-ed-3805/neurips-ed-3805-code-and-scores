import os, json, csv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_RES = os.path.join(_ROOT, "reports")  

OUT = os.path.join(FINAL_RES, "non_oracle_gains")

JUDGES = ["gemma2_9b", "gemma3_27b", "qwen3_4b", "mistral_7b", "llama31_8b", "llama31_70b", "gemini2_flash", "gemini3_flash", "gemini31_pro", "gpt54", "opus47"]

MPP = {
    "gemma2_9b": "Gemma-2-9B", "gemma3_27b": "Gemma-3-27B", "qwen3_4b": "Qwen3-4B",
    "mistral_7b": "Mistral-7B", "llama31_8b": "LLaMA-3.1-8B", "llama31_70b": "LLaMA-3.1-70B",
    "gemini2_flash": "Gemini-2.0-Flash", "gemini3_flash": "Gemini-3-Flash",
    "gemini31_pro": "Gemini 3.1 Pro", "gpt54": "GPT 5.4", "opus47": "Claude Opus 4.7",
}

FRACS = ["0.1", "0.2", "0.3", "0.4", "0.5"]
DATASETS = ["summeval", "newsroom", "tc", "wp_a"]


def load():
    data = {}
    for j in JUDGES:
        fp = os.path.join(OUT, "per_judge", f"{j}.json")
        if os.path.exists(fp):
            data[j] = json.load(open(fp))
    return data


def main():
    data = load()
    if not data:
        print("No per-judge json found. Run the sweep first."); return

    # calibration_gains_by_metric.csv
    with open(os.path.join(OUT, "calibration_gains_by_metric.csv"), "w", newline="") as f:

        w = csv.writer(f)
        w.writerow(["judge", "dataset", "metric", "fraction", "sel_spearman", "canon_spearman", "delta_spearman", "test_oracle_regret", "pick_canonical_rate", "canonical_range", "calib_size", "n_units"])
        
        for j in JUDGES:
            if j not in data:
                continue
            for ds in DATASETS:
                mets = data[j]["datasets"].get(ds, {}).get("metrics", {})
                for m, byfrac in mets.items():
                    for fr in FRACS:
                        if fr not in byfrac:
                            continue
                        a = byfrac[fr]
                        w.writerow([MPP[j], ds, m, fr, f"{a['sel_spearman']:.4f}",
                                    f"{a['canon_spearman']:.4f}", f"{a['delta_spearman']:+.4f}",
                                    f"{a['test_oracle_regret']:.4f}", f"{a['pick_canonical_rate']:.3f}",
                                    a["canonical_range"], a["calib_size"], a["n_units"]])

    # calibration_gains_overall.csv
    with open(os.path.join(OUT, "calibration_gains_overall.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["judge"]

        for fr in FRACS:
            head += [f"sel@{fr}", f"can@{fr}", f"d@{fr}"]
        w.writerow(head)
        for j in JUDGES:
            if j not in data:
                continue
            row = [MPP[j]]
            ov = data[j]["overall"]
            for fr in FRACS:
                if fr in ov:
                    row += [f"{ov[fr]['sel_spearman']:.3f}", f"{ov[fr]['canon_spearman']:.3f}",
                            f"{ov[fr]['delta_spearman']:+.3f}"]
                else:
                    row += ["", "", ""]
            w.writerow(row)

    # calibration_gains_by_dataset.csv
    with open(os.path.join(OUT, "calibration_gains_by_dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "dataset"] + [f"sel@{fr}" for fr in FRACS] + [f"can@{fr}" for fr in FRACS] + [f"d@{fr}" for fr in FRACS])

        for j in JUDGES:
            if j not in data:
                continue

            for ds in DATASETS:
                avg = data[j]["datasets"].get(ds, {}).get("average", {})
                if not avg:
                    continue
                
                sels = [f"{avg[fr]['sel_spearman']:.3f}" if fr in avg else "" for fr in FRACS]
                cans = [f"{avg[fr]['canon_spearman']:.3f}" if fr in avg else "" for fr in FRACS]
                dels = [f"{avg[fr]['delta_spearman']:+.3f}" if fr in avg else "" for fr in FRACS]
                w.writerow([MPP[j], ds] + sels + cans + dels)

    print(f"Wrote calibration_gains_by_metric.csv, calibration_gains_overall.csv, " f"calibration_gains_by_dataset.csv to {OUT}")


if __name__ == "__main__":
    main()
