"""Reproduce the numerical results supporting Two Routes to Quantum Tunnelling.

Usage:
    python reproduce.py recorded
    python reproduce.py estimate
    python reproduce.py run --quick
    python reproduce.py run

Outputs JSON to standard output; saved historical data are never overwritten.
The full original run includes L=24 and may take a few minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import original
import parity


DATA = Path(__file__).resolve().parent / "data"


def recorded(name):
    with (DATA / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def runtime_estimate():
    """Illustrative extrapolation, not a measured runtime or a lower bound."""
    anchor = next(row for row in recorded("original") if int(row["L"]) == 24)
    seconds_per_step = float(anchor["route1_wall_seconds"]) / int(anchor["route1_steps"])
    reference = original.wronskian_reference(100, 0.5)
    log_steps = original.estimate_uniform_steps(reference.log_gap, 100)
    seconds = math.exp(log_steps) * seconds_per_step
    return {
        "L": 100,
        "estimated_steps": math.exp(log_steps),
        "assumed_seconds_per_step": seconds_per_step,
        "estimated_seconds": seconds,
        "estimated_years": seconds / (365.25 * 86400),
        "assumption": "Original step-count model, keeping the measured L=24 throughput fixed.",
        "limitation": "Extrapolated, not run; no optimisation to a common 1% error target."
    }


def original_row(L):
    _, _, reference = original.exact_reference(L, 0.5)
    result = original.uniform_ite_gap(L, 0.5)
    return {
        "L": L,
        "gap": result.gap,
        "reference_gap": reference,
        "relative_error": abs(result.gap / reference - 1.0),
        "steps": result.steps,
        "new_runtime_seconds": result.wall_seconds,
    }


def parity_row(L):
    problem = parity.build_problem(L, 0.5, "coherent")
    # Preserve the original benchmark's per-block Gershgorin bound and its
    # reference-energy-scaled floating-point margin. The latter is disclosed
    # in README.md; this is a benchmark, not a reference-free stopping method.
    shifts = tuple(
        parity.gershgorin_bounds(sector.diag, sector.off)[0]
        - 32.0 * np.finfo(float).eps * max(1.0, abs(sector.energy0))
        for sector in (problem.even, problem.odd)
    )
    history = parity.run_uniform(
        problem, "uniform-cn", shifts, (1e-10,), dt_factor=2.0,
        check_every=5, max_pair_steps=1_000_000,
    )
    hits = [row for row in history if row["relative_gap_error"] <= 1e-10]
    if not hits:
        raise RuntimeError(f"Target not reached for L={L}")
    hit = hits[0]
    return {
        "L": L,
        "log_gap": hit["log_gap"],
        "relative_error": hit["relative_gap_error"],
        "sector_solves": hit["total_sector_steps"],
        "new_runtime_seconds": hit["elapsed_seconds"],
    }


def instanton_rows():
    result = []
    for row in recorded("instanton"):
        L = int(row["L"])
        if L <= 24:
            _, _, reference = original.exact_reference(L, 0.5)
            error_percent = abs(original.instanton_gap(L, 0.5) / reference - 1.0) * 100
        else:
            reference = original.wronskian_reference(L, 0.5)
            error_percent = abs(math.expm1(
                original.instanton_log_gap(L, 0.5) - reference.log_gap
            )) * 100
        result.append({"L": L, "route2_error_percent": error_percent})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("recorded", "estimate", "run"))
    parser.add_argument("--quick", action="store_true", help="Run original ITE only at L=8.")
    args = parser.parse_args()
    if args.command == "recorded":
        result = {name: recorded(name) for name in ("original", "instanton", "parity")}
    elif args.command == "estimate":
        result = runtime_estimate()
    else:
        result = {
            "original": [original_row(L) for L in ([8] if args.quick else [8, 16, 20, 24])],
            "parity": [parity_row(L) for L in (160, 800, 2800)],
            "instanton": instanton_rows(),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
