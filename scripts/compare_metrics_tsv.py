#!/usr/bin/env python3
"""Compare two OrganSMNIST InfoGAN runs from structured metrics.tsv logs.

Usage:
  python scripts/compare_metrics_tsv.py \
    --control runs/ca_off/metrics.tsv \
    --treatment runs/ca_on/metrics.tsv \
    --iter-min 1000 --iter-max 12000
"""

import argparse
import csv
import math
from typing import Dict, List, Optional


def _to_float(value: Optional[str]) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    if text.lower() == "inf":
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _mean(values: List[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return float("nan")
    return sum(valid) / float(len(valid))


def _fraction(values: List[float], predicate) -> float:
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return float("nan")
    hits = sum(1 for v in valid if predicate(v))
    return hits / float(len(valid))


def load_rows(path: str, iter_min: int, iter_max: Optional[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for raw in reader:
            it = int(_to_float(raw.get("iter")))
            if it < iter_min:
                continue
            if iter_max is not None and it > iter_max:
                continue
            row = {k: _to_float(v) for k, v in raw.items() if k is not None}
            row["iter"] = float(it)
            rows.append(row)
    return rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    cols = lambda key: [r.get(key, float("nan")) for r in rows]
    adv_max = cols("adv_max")
    mi_avg = cols("mi_avg")
    r_sense = cols("r_sense_avg")
    r_intra = cols("r_intra_avg")
    r_shape_div = cols("r_shape_div_avg")
    r_shape_div_min = cols("r_shape_div_min_avg")
    rfl = cols("real_fake_loss")
    ca_blend = cols("ca_blend")

    return {
        "n_rows": float(len(rows)),
        "adv_max_mean": _mean(adv_max),
        "adv_max_gt_-0.6": _fraction(adv_max, lambda v: v > -0.6),
        "adv_max_le_-0.6": _fraction(adv_max, lambda v: v <= -0.6),
        "mi_avg_mean": _mean(mi_avg),
        "r_sense_mean": _mean(r_sense),
        "r_intra_mean": _mean(r_intra),
        "r_shape_div_mean": _mean(r_shape_div),
        "r_shape_div_min_mean": _mean(r_shape_div_min),
        "rfl_mean": _mean(rfl),
        "rfl_0.4_0.6": _fraction(rfl, lambda v: 0.4 <= v <= 0.6),
        "rfl_lt_0.4": _fraction(rfl, lambda v: v < 0.4),
        "rfl_gt_0.6": _fraction(rfl, lambda v: v > 0.6),
        "ca_blend_mean": _mean(ca_blend),
    }


def fmt(value: float, pct: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    if pct:
        return f"{100.0 * value:.1f}%"
    return f"{value:.4f}"


def print_summary(name: str, summary: Dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  rows: {int(summary['n_rows'])}")
    print(f"  adv_max mean: {fmt(summary['adv_max_mean'])}")
    print(f"  adv_max > -0.6: {fmt(summary['adv_max_gt_-0.6'], pct=True)}")
    print(f"  adv_max <= -0.6: {fmt(summary['adv_max_le_-0.6'], pct=True)}")
    print(f"  mi_avg mean: {fmt(summary['mi_avg_mean'])}")
    print(f"  r_sense mean: {fmt(summary['r_sense_mean'])}")
    print(f"  r_intra mean: {fmt(summary['r_intra_mean'])}")
    print(f"  r_shape_div mean: {fmt(summary['r_shape_div_mean'])}")
    print(f"  r_shape_div_min mean: {fmt(summary['r_shape_div_min_mean'])}")
    print(f"  real_fake_loss mean: {fmt(summary['rfl_mean'])}")
    print(f"  real_fake_loss in [0.4,0.6]: {fmt(summary['rfl_0.4_0.6'], pct=True)}")
    print(f"  real_fake_loss < 0.4: {fmt(summary['rfl_lt_0.4'], pct=True)}")
    print(f"  real_fake_loss > 0.6: {fmt(summary['rfl_gt_0.6'], pct=True)}")
    print(f"  ca_blend mean: {fmt(summary['ca_blend_mean'])}")


def print_delta(control: Dict[str, float], treatment: Dict[str, float]) -> None:
    print("\nTreatment - Control")
    keys = [
        "adv_max_mean",
        "adv_max_gt_-0.6",
        "mi_avg_mean",
        "r_sense_mean",
        "r_intra_mean",
        "r_shape_div_mean",
        "r_shape_div_min_mean",
        "rfl_mean",
        "rfl_0.4_0.6",
        "rfl_lt_0.4",
        "rfl_gt_0.6",
        "ca_blend_mean",
    ]
    for k in keys:
        c = control.get(k, float("nan"))
        t = treatment.get(k, float("nan"))
        d = t - c if math.isfinite(c) and math.isfinite(t) else float("nan")
        if k.startswith("rfl_") or k.startswith("adv_max_gt_"):
            print(f"  {k}: {fmt(d, pct=True)}")
        else:
            print(f"  {k}: {fmt(d)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, help="Path to control metrics.tsv")
    parser.add_argument("--treatment", required=True, help="Path to treatment metrics.tsv")
    parser.add_argument("--iter-min", type=int, default=0, help="Minimum iteration to include")
    parser.add_argument("--iter-max", type=int, default=None, help="Maximum iteration to include")
    args = parser.parse_args()

    control_rows = load_rows(args.control, args.iter_min, args.iter_max)
    treatment_rows = load_rows(args.treatment, args.iter_min, args.iter_max)

    if not control_rows:
        raise SystemExit("No rows loaded for control run in selected window.")
    if not treatment_rows:
        raise SystemExit("No rows loaded for treatment run in selected window.")

    control_summary = summarize(control_rows)
    treatment_summary = summarize(treatment_rows)

    print(f"Window: iter >= {args.iter_min}" + ("" if args.iter_max is None else f", iter <= {args.iter_max}"))
    print_summary("Control", control_summary)
    print_summary("Treatment", treatment_summary)
    print_delta(control_summary, treatment_summary)


if __name__ == "__main__":
    main()
