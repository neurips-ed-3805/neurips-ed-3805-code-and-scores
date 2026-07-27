#!/usr/bin/env python3
"""
Cross-range correlations for frontier judges over the standard ranges

Models:
- gpt54          (GPT 5.4)
- gemini31_pro   (Gemini 3.1 Pro)
- opus47         (Claude Opus 4.7)

Ranges: [0,1], [1,3], [1,5], [0,10], [0,100]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

MODEL_DIRS = {
    "gpt54": "GPT 5.4",
    "gemini31_pro": "Gemini 3.1 Pro",
    "opus47": "Claude Opus 4.7",
}

STANDARD_RANGES: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (1.0, 3.0),
    (1.0, 5.0),
    (0.0, 10.0),
    (0.0, 100.0),
]


def range_label(r: Tuple[float, float]) -> str:
    return f"[{int(r[0]) if r[0].is_integer() else r[0]},{int(r[1]) if r[1].is_integer() else r[1]}]"


def range_file_name(r: Tuple[float, float]) -> str:
    a = str(int(r[0])) if r[0].is_integer() else str(r[0]).replace(".", "_")
    b = str(int(r[1])) if r[1].is_integer() else str(r[1]).replace(".", "_")
    return f"range_{a}_{b}.json"


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_first_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list):
        if not x:
            return None
        return parse_first_float(x[0])
    if isinstance(x, str):
        m = FLOAT_RE.search(x.strip())
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def stable_hash(parts: Iterable[str]) -> str:
    s = "||".join(parts)
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_01(v: float, rmin: float, rmax: float) -> float:
    if rmax == rmin:
        return math.nan
    z = (v - rmin) / (rmax - rmin)
    return float(np.clip(z, 0.0, 1.0))


@dataclass
class Extracted:
    instance_key: str
    item_key: str
    score: float


def extract_records(data: list, dataset: str) -> List[Extracted]:
    out: List[Extracted] = []

    for i, rec in enumerate(data):

        if isinstance(rec, dict) and "all_responses" in rec:
            raw = parse_first_float(rec.get("all_responses"))
            if raw is None:
                continue

            # WritingPrompts-A rows are one story per prompt, so per-instance
            if dataset == "wp_a" and "story" in rec:
                instance_key = "__global__"
                item_key = "wpstory::" + stable_hash([
                    str(rec.get("story", "")),
                    str(rec.get("human_written", "")),
                ])
                out.append(Extracted(instance_key=instance_key, item_key=item_key, score=raw))
                continue

            if "doc_id" in rec and "system_id" in rec:
                instance_key = f"doc::{rec.get('doc_id')}"
                item_key = f"sys::{rec.get('system_id')}"
            elif "doc_id" in rec and "system_output" in rec:
                instance_key = f"doc::{rec.get('doc_id')}"
                item_key = "sysout::" + stable_hash([
                    str(rec.get("system_output", "")),
                ])
            elif "doc_id" in rec and "model" in rec:
                instance_key = f"doc::{rec.get('doc_id')}"
                item_key = f"model::{rec.get('model')}"
            elif "prompt" in rec and "story" in rec:
                instance_key = "__global__"
                item_key = "wp::" + stable_hash([
                    str(rec.get("story", "")),
                    str(rec.get("human_written", "")),
                ])
            else:
                # Fallback
                instance_key = "__global__"
                item_key = "row::" + stable_hash([
                    str(rec.get("doc_id", "")),
                    str(rec.get("system_id", "")),
                    str(rec.get("model", "")),
                    str(rec.get("source", "")),
                    str(rec.get("system_output", "")),
                    str(rec.get("story", "")),
                    str(i),
                ])

            out.append(Extracted(instance_key=instance_key, item_key=item_key, score=raw))
            continue

        if isinstance(rec, dict) and isinstance(rec.get("responses"), list):
            top_context = str(rec.get("context", ""))
            top_fact = str(rec.get("fact", ""))
            for j, rsp in enumerate(rec["responses"]):
                if not isinstance(rsp, dict):
                    continue
                raw = parse_first_float(rsp.get("all_responses"))
                if raw is None:
                    continue
                instance_key = "tcinst::" + stable_hash([
                    top_context,
                    top_fact,
                ])
                item_key = "tcresp::" + stable_hash([
                    str(rsp.get("model", "")),
                    str(rsp.get("response", "")),
                    str(j),
                ])
                out.append(Extracted(instance_key=instance_key, item_key=item_key, score=raw))

    return out


def avg_instance_corr(
    maps_i: Dict[str, Dict[str, float]],
    maps_j: Dict[str, Dict[str, float]],
    method: str,
) -> Tuple[float, int]:
    vals = []
    common_instances = set(maps_i.keys()) & set(maps_j.keys())
    for inst in common_instances:
        items = set(maps_i[inst].keys()) & set(maps_j[inst].keys())
        if len(items) < 2:
            continue
        x = np.array([maps_i[inst][k] for k in sorted(items)], dtype=float)
        y = np.array([maps_j[inst][k] for k in sorted(items)], dtype=float)
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        try:
            if method == "pearson":
                c = pearsonr(x, y)[0]
            elif method == "spearman":
                c = spearmanr(x, y).correlation
            else:
                raise ValueError(method)
        except Exception:
            c = np.nan
        if not np.isnan(c):
            vals.append(float(c))
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)


def corr_matrix(range_instance_maps: List[Dict[str, Dict[str, float]]], method: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(range_instance_maps)
    mat = np.eye(n, dtype=float)
    cnt = np.zeros((n, n), dtype=int)
    np.fill_diagonal(cnt, -1)
    for i in range(n):
        for j in range(i + 1, n):
            c, ninst = avg_instance_corr(range_instance_maps[i], range_instance_maps[j], method)
            mat[i, j] = c
            mat[j, i] = c
            cnt[i, j] = ninst
            cnt[j, i] = ninst
    return mat, cnt


def run(base_scores_dir: Path) -> None:
    labels = [range_label(r) for r in STANDARD_RANGES]
    summary_rows = []

    for model_dir, model_name in MODEL_DIRS.items():
        model_root = base_scores_dir / model_dir
        if not model_root.exists():
            print(f"[WARN] Missing model directory: {model_root}")
            continue

        datasets = sorted([p.name for p in model_root.iterdir() if p.is_dir()])
        for dataset in datasets:
            ds_root = model_root / dataset
            metrics = sorted([p.name for p in ds_root.iterdir() if p.is_dir()])

            for metric in metrics:
                metric_root = ds_root / metric

                # Load + normalize 5 ranges into instance->item->score
                range_maps: List[Dict[str, Dict[str, float]]] = []
                missing = False
                for r in STANDARD_RANGES:
                    fp = metric_root / range_file_name(r)
                    if not fp.exists():
                        print(f"[WARN] Missing file: {fp}")
                        missing = True
                        break
                    try:
                        data = json.loads(fp.read_text())
                    except Exception as e:
                        print(f"[WARN] Failed to load {fp}: {e}")
                        missing = True
                        break

                    extracted = extract_records(data, dataset=dataset)
                    if not extracted:
                        print(f"[WARN] No scores extracted from {fp}")
                        missing = True
                        break

                    rm: Dict[str, Dict[str, float]] = {}
                    for e in extracted:
                        rm.setdefault(e.instance_key, {})[e.item_key] = normalize_01(e.score, r[0], r[1])
                    range_maps.append(rm)

                if missing:
                    continue

                pear, pear_n = corr_matrix(range_maps, method="pearson")
                spear, spear_n = corr_matrix(range_maps, method="spearman")

                # rough instance count (intersection over all ranges)
                common_instances = set(range_maps[0].keys())
                for rm in range_maps[1:]:
                    common_instances &= set(rm.keys())

                row = {
                    "model": model_dir,
                    "dataset": dataset,
                    "metric": metric,
                    "n_instances_common_all_ranges": len(common_instances),
                    "correlation_method": "per_instance_avg",
                }

                for i in range(len(STANDARD_RANGES)):
                    for j in range(i + 1, len(STANDARD_RANGES)):
                        pair = f"{labels[i]}__{labels[j]}"
                        row[f"n_instances_pearson_{pair}"] = int(pear_n[i, j])
                        row[f"n_instances_spearman_{pair}"] = int(spear_n[i, j])
                        row[f"pearson_{pair}"] = float(pear[i, j]) if not np.isnan(pear[i, j]) else np.nan
                        row[f"spearman_{pair}"] = float(spear[i, j]) if not np.isnan(spear[i, j]) else np.nan

                summary_rows.append(row)
                print(f"[OK] {model_dir}/{dataset}/{metric} | instances_common={len(common_instances)}")


    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = reports_dir / "cross_range_correlation_summary.csv"

    df = pd.DataFrame(summary_rows)
    if not df.empty:
        df = df.sort_values(["model", "dataset", "metric"]).reset_index(drop=True)
    df.to_csv(summary_csv, index=False)

    print("\nDone.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Total tasks: {len(summary_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "scores",
        help="Root directory containing model score folders",
    )
    args = parser.parse_args()

    run(base_scores_dir=args.scores_dir)


if __name__ == "__main__":
    main()
