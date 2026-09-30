"""Parity-resolved uniform-CN benchmark and Wronskian readout.

Relevant functions extracted without changing their numerical logic.
The public runner selects coherent states, the Gershgorin lower shift,
conservative uniform CN and reference-graded first-hit error = 1e-10.
The generic theta-step helper is retained to preserve the original arithmetic;
only theta = 1/2 (CN) is used by this package's runner.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.special import gammaln


def parity_blocks(L: int, gamma: float):
    """Derive the even and odd blocks in the non-negative-m basis."""
    if L <= 0 or L % 2:
        raise ValueError("L must be a positive even integer")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie between zero and one")

    S = L // 2
    m_even = np.arange(0, S + 1, dtype=float)
    d_even = -2.0 * m_even**2 / L
    o_even = -gamma * np.sqrt(
        (L / 2.0) * (L / 2.0 + 1.0)
        - m_even[:-1] * (m_even[:-1] + 1.0)
    )
    o_even[0] *= math.sqrt(2.0)

    m_odd = np.arange(1, S + 1, dtype=float)
    d_odd = -2.0 * m_odd**2 / L
    o_odd = -gamma * np.sqrt(
        (L / 2.0) * (L / 2.0 + 1.0)
        - m_odd[:-1] * (m_odd[:-1] + 1.0)
    )
    return (d_even, o_even), (d_odd, o_odd)


def apply_tridiagonal(diag: np.ndarray, off: np.ndarray,
                      state: np.ndarray) -> np.ndarray:
    out = diag * state
    out[1:] += off * state[:-1]
    out[:-1] += off * state[1:]
    return out


def normalise(state: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(state))
    if not math.isfinite(norm) or norm == 0.0:
        raise FloatingPointError("state acquired a zero or non-finite norm")
    return state / norm


def rayleigh(diag: np.ndarray, off: np.ndarray,
             state: np.ndarray) -> float:
    return float(np.dot(state, apply_tridiagonal(diag, off, state)))


def residual_norm(diag: np.ndarray, off: np.ndarray,
                  state: np.ndarray, energy: float) -> float:
    return float(np.linalg.norm(
        apply_tridiagonal(diag, off, state) - energy * state
    ))


def gershgorin_bounds(diag: np.ndarray, off: np.ndarray):
    radius = np.zeros(diag.size)
    radius[:-1] += np.abs(off)
    radius[1:] += np.abs(off)
    return float(np.min(diag - radius)), float(np.max(diag + radius))


def coherent_amplitudes(L: int, gamma: float) -> np.ndarray:
    """Positive-m classical spin-coherent product state."""
    q = math.sqrt(1.0 - gamma**2)
    p = 0.5 * (1.0 + q)
    k = np.arange(L + 1, dtype=float)
    log_amp = 0.5 * (
        gammaln(L + 1.0) - gammaln(k + 1.0) - gammaln(L - k + 1.0)
        + k * math.log(p) + (L - k) * math.log1p(-p)
    )
    log_amp -= float(np.max(log_amp))
    return normalise(np.exp(log_amp))


def parity_projected_initial(L: int, gamma: float, parity: str,
                             kind: str = "coherent") -> np.ndarray:
    S = L // 2
    if kind == "central":
        size = S + 1 if parity == "even" else S
        return normalise(np.exp(
            -0.5 * np.arange(size, dtype=float) ** 2 / L
        ))
    if kind != "coherent":
        raise ValueError(f"unknown initial-state kind {kind!r}")

    full = coherent_amplitudes(L, gamma)
    if parity == "even":
        state = np.empty(S + 1)
        state[0] = full[S]
        for m in range(1, S + 1):
            state[m] = (full[S + m] + full[S - m]) / math.sqrt(2.0)
    elif parity == "odd":
        state = np.empty(S)
        for m in range(1, S + 1):
            state[m - 1] = (
                full[S + m] - full[S - m]
            ) / math.sqrt(2.0)
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    return normalise(state)


def wronskian_log(L: int, gamma: float, even: np.ndarray,
                  odd: np.ndarray) -> float:
    """Exact discrete-spin Herring/Wronskian expression in log form."""
    denominator = abs(float(np.dot(even[1:], odd)))
    hopping = gamma * math.sqrt((L / 2.0) * (L / 2.0 + 1.0))
    if denominator == 0.0 or even[0] == 0.0 or odd[0] == 0.0:
        return float("-inf")
    return float(
        math.log(math.sqrt(2.0) * hopping)
        + math.log(abs(float(even[0])))
        + math.log(abs(float(odd[0])))
        - math.log(denominator)
    )


def relative_log_error(log_value: float, log_reference: float) -> float:
    if not math.isfinite(log_value):
        return float("inf")
    delta = log_value - log_reference
    # Avoid a noisy overflow warning when a deliberately over-long CN sweep
    # has already destroyed the state.  At that point the relative error is
    # unambiguously infinite for the purposes of this audit.
    if delta > math.log(np.finfo(float).max):
        return float("inf")
    return abs(float(np.expm1(delta)))


@dataclass
class Sector:
    parity: str
    diag: np.ndarray
    off: np.ndarray
    energy0: float
    energy1: float
    energy_max: float
    exact_state: np.ndarray
    initial_state: np.ndarray
    initial_overlap: float
    initial_mu: float
    initial_residual: float

    @property
    def same_sector_gap(self) -> float:
        return self.energy1 - self.energy0


@dataclass
class Problem:
    L: int
    gamma: float
    even: Sector
    odd: Sector
    reference_log_gap: float


def build_problem(L: int, gamma: float, initial_kind: str) -> Problem:
    sectors = []
    for parity, (diag, off) in zip(
        ("even", "odd"), parity_blocks(L, gamma)
    ):
        energies, vectors = eigh_tridiagonal(
            diag, off, select="i", select_range=(0, 1),
            lapack_driver="stemr", check_finite=False
        )
        energy_max = float(eigh_tridiagonal(
            diag, off, select="i", select_range=(diag.size - 1,
                                                   diag.size - 1),
            eigvals_only=True, lapack_driver="stemr",
            check_finite=False,
        )[0])
        exact = normalise(vectors[:, 0])
        initial = parity_projected_initial(
            L, gamma, parity, initial_kind
        )
        mu = rayleigh(diag, off, initial)
        residual = residual_norm(diag, off, initial, mu)
        sectors.append(Sector(
            parity=parity,
            diag=diag,
            off=off,
            energy0=float(energies[0]),
            energy1=float(energies[1]),
            energy_max=energy_max,
            exact_state=exact,
            initial_state=initial,
            initial_overlap=abs(float(np.dot(initial, exact))),
            initial_mu=mu,
            initial_residual=residual,
        ))
    even, odd = sectors
    return Problem(
        L=L,
        gamma=gamma,
        even=even,
        odd=odd,
        reference_log_gap=wronskian_log(
            L, gamma, even.exact_state, odd.exact_state
        ),
    )


def base_step(sector: Sector, shift: float,
              dt_factor: float = 2.0) -> float:
    _, upper = gershgorin_bounds(sector.diag - shift, sector.off)
    if upper <= 0.0:
        raise FloatingPointError("non-positive shifted spectral upper bound")
    return dt_factor / upper


def make_theta_stepper(sector: Sector, shift: float, dtau: float,
                       theta: float):
    """Scaled implicit theta step; theta=1/2 is CN and theta=1 is BE."""
    if theta not in (0.5, 1.0):
        raise ValueError("theta must be 0.5 or 1")
    if dtau <= 0.0 or not math.isfinite(dtau):
        raise ValueError("dtau must be finite and positive")

    diag = sector.diag - shift
    off = sector.off
    alpha = 1.0 / (theta * dtau)
    beta = (1.0 - theta) / theta

    lhs = np.zeros((3, diag.size))
    lhs[0, 1:] = off
    lhs[1] = diag + alpha
    lhs[2, :-1] = off
    right_diag = alpha - beta * diag
    right_off = -beta * off

    def step(state: np.ndarray) -> np.ndarray:
        rhs = right_diag * state
        rhs[1:] += right_off * state[:-1]
        rhs[:-1] += right_off * state[1:]
        return normalise(solve_banded(
            (1, 1), lhs, rhs, check_finite=False
        ))

    return step


def checkpoint(problem: Problem, states: list[np.ndarray], pair_steps: int,
               elapsed: float, panel: int | None,
               dtau_even: float, dtau_odd: float,
               tau_even: float, tau_odd: float):
    log_gap = wronskian_log(problem.L, problem.gamma, states[0], states[1])
    mus = [
        rayleigh(sector.diag, sector.off, state)
        for sector, state in zip((problem.even, problem.odd), states)
    ]
    residuals = [
        residual_norm(sector.diag, sector.off, state, mu)
        for sector, state, mu in zip(
            (problem.even, problem.odd), states, mus
        )
    ]
    return {
        "pair_steps": pair_steps,
        "total_sector_steps": 2 * pair_steps,
        "elapsed_seconds": elapsed,
        "panel": panel,
        "dtau_even": dtau_even,
        "dtau_odd": dtau_odd,
        "tau_even": tau_even,
        "tau_odd": tau_odd,
        "log_gap": log_gap,
        "relative_gap_error": relative_log_error(
            log_gap, problem.reference_log_gap
        ),
        "even_overlap": abs(float(np.dot(
            states[0], problem.even.exact_state
        ))),
        "odd_overlap": abs(float(np.dot(
            states[1], problem.odd.exact_state
        ))),
        "even_rayleigh": mus[0],
        "odd_rayleigh": mus[1],
        "even_energy_error": mus[0] - problem.even.energy0,
        "odd_energy_error": mus[1] - problem.odd.energy0,
        "even_residual": residuals[0],
        "odd_residual": residuals[1],
    }


def run_uniform(problem: Problem, method: str, shifts: tuple[float, float],
                thresholds: Sequence[float], dt_factor: float,
                check_every: int, max_pair_steps: int,
                absolute_dtau: float | None = None):
    theta = 0.5 if method == "uniform-cn" else 1.0
    sectors = (problem.even, problem.odd)
    dtaus = (
        (absolute_dtau, absolute_dtau)
        if absolute_dtau is not None
        else tuple(
            base_step(sector, shift, dt_factor)
            for sector, shift in zip(sectors, shifts)
        )
    )
    states = [sector.initial_state.copy() for sector in sectors]
    tau = [0.0, 0.0]
    started = time.perf_counter()
    steppers = [
        make_theta_stepper(sector, shift, dtau, theta)
        for sector, shift, dtau in zip(sectors, shifts, dtaus)
    ]
    history = [checkpoint(
        problem, states, 0, time.perf_counter() - started, None,
        dtaus[0], dtaus[1], tau[0], tau[1]
    )]
    strictest = min(thresholds)
    confirmation_checks = 0

    for pair_step in range(1, max_pair_steps + 1):
        states[0] = steppers[0](states[0])
        states[1] = steppers[1](states[1])
        tau[0] += dtaus[0]
        tau[1] += dtaus[1]
        if pair_step % check_every and pair_step != max_pair_steps:
            continue
        row = checkpoint(
            problem, states, pair_step, time.perf_counter() - started,
            None, dtaus[0], dtaus[1], tau[0], tau[1]
        )
        history.append(row)
        if row["relative_gap_error"] <= strictest:
            confirmation_checks += 1
            if confirmation_checks >= 4:
                break
        else:
            confirmation_checks = 0

    return history
