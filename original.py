"""Original uniform-CN benchmark and leading instanton formula.

Only the numerical components needed for the poster are included.
Units: interaction strength = hbar = 1; Gamma = 0.5 in the reported data.
The original uniform solver does not separate parity sectors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded


@dataclass(frozen=True)
class ITEResult:
    """Result of one uniform imaginary-time calculation."""

    gap: float
    steps: int
    tau: float
    wall_seconds: float
    upper_magnetisation: float
    lower_magnetisation: float


@dataclass(frozen=True)
class WronskianReference:
    """Cancellation-free stationary reference for the doublet gap."""

    gap: float
    log_gap: float
    overlap_denominator: float


def collective_hamiltonian(L: int, gamma: float):
    """Return the tridiagonal collective-spin Hamiltonian and m values.

    ``L`` is the number of spin-1/2 sites and must be a positive even integer.
    ``gamma`` is the dimensionless transverse field.  The broken-symmetry
    tunnelling regime is 0 < gamma < 1.
    """
    if L <= 0 or L % 2:
        raise ValueError("L must be a positive even integer")
    if not 0.0 < gamma < 1.0:
        raise ValueError("the tunnelling formula requires 0 < gamma < 1")

    S = L / 2.0
    m = np.arange(-S, S + 1.0)
    diag = -2.0 * m**2 / L
    off = -gamma * np.sqrt(S * (S + 1.0) - m[:-1] * (m[:-1] + 1.0))
    return diag, off, m


def apply_tridiagonal(diag, off, vector):
    """Apply a real symmetric tridiagonal matrix to a vector."""
    result = diag * vector
    result[1:] += off * vector[:-1]
    result[:-1] += off * vector[1:]
    return result


def exact_reference(L: int, gamma: float):
    """Return E0, E1 and their gap from direct diagonalisation."""
    diag, off, _ = collective_hamiltonian(L, gamma)
    energies = eigh_tridiagonal(
        diag,
        off,
        eigvals_only=True,
        select="i",
        select_range=(0, 1),
        check_finite=False,
    )
    return float(energies[0]), float(energies[1]), float(energies[1] - energies[0])


def parity_blocks(L: int, gamma: float):
    """Return even and odd tridiagonal blocks of the collective Hamiltonian.

    For even ``L``, the even basis is ``|0>`` followed by
    ``(|m>+|-m>)/sqrt(2)``, while the odd basis is
    ``(|m>-|-m>)/sqrt(2)`` for ``m=1,...,S``.
    """
    collective_hamiltonian(L, gamma)  # validate the inputs
    S = L // 2

    even_m = np.arange(0, S + 1, dtype=float)
    even_diag = -2.0 * even_m**2 / L
    even_off = -gamma * np.sqrt(
        (L / 2.0) * (L / 2.0 + 1.0)
        - even_m[:-1] * (even_m[:-1] + 1.0)
    )
    even_off[0] *= np.sqrt(2.0)

    odd_m = np.arange(1, S + 1, dtype=float)
    odd_diag = -2.0 * odd_m**2 / L
    odd_off = -gamma * np.sqrt(
        (L / 2.0) * (L / 2.0 + 1.0)
        - odd_m[:-1] * (odd_m[:-1] + 1.0)
    )
    return even_diag, even_off, odd_diag, odd_off


def wronskian_reference(L: int, gamma: float):
    """Return the gap from an exact discrete Wronskian identity.

    Let ``a`` and ``b`` be the lowest eigenvectors of the even and odd parity
    blocks.  Discrete summation by parts gives

        gap = -sqrt(2) t0 a[0] b[0] / dot(a[1:], b),

    where ``t0`` is the full-basis hopping between ``m=0`` and ``m=1``.
    This is exact for exact eigenvectors and avoids subtracting the nearly equal
    even and odd energies.  The logarithmic form remains useful after the gap
    itself becomes too small to represent as a normal double.

    The MRRR ``stemr`` driver is selected explicitly.  Other LAPACK drivers
    were observed to lose the exponentially small central tail earlier.
    """
    even_diag, even_off, odd_diag, odd_off = parity_blocks(L, gamma)
    _, even_vectors = eigh_tridiagonal(
        even_diag,
        even_off,
        select="i",
        select_range=(0, 0),
        lapack_driver="stemr",
        check_finite=False,
    )
    _, odd_vectors = eigh_tridiagonal(
        odd_diag,
        odd_off,
        select="i",
        select_range=(0, 0),
        lapack_driver="stemr",
        check_finite=False,
    )
    even = even_vectors[:, 0]
    odd = odd_vectors[:, 0]
    denominator = abs(float(np.dot(even[1:], odd)))
    hopping_m0 = -gamma * np.sqrt((L / 2.0) * (L / 2.0 + 1.0))

    if even[0] == 0.0 or odd[0] == 0.0:
        raise FloatingPointError(
            "central eigenvector tail underflowed; use arbitrary precision"
        )
    log_gap = (
        np.log(np.sqrt(2.0) * abs(hopping_m0))
        + np.log(abs(even[0]))
        + np.log(abs(odd[0]))
        - np.log(denominator)
    )
    gap = float(np.exp(log_gap))
    return WronskianReference(gap, float(log_gap), denominator)


def instanton_action_density(gamma: float):
    """Return c(Gamma), for which the large-L gap scales as exp(-L c)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("0 < gamma < 1 is required")
    q = np.sqrt(1.0 - gamma**2)
    return 0.5 * np.log((1.0 + q) / (1.0 - q)) - q


def instanton_log_gap(L: int, gamma: float, prefactor: str = "garg"):
    """Return log(gap), avoiding underflow at very large ``L``."""
    q = np.sqrt(1.0 - gamma**2)
    if prefactor == "garg":
        power_base = 1.0 - gamma**2
    elif prefactor == "isakov-printed":
        power_base = 1.0 - gamma
    else:
        raise ValueError("prefactor must be 'garg' or 'isakov-printed'")
    return float(
        0.5 * np.log(32.0 * L / np.pi)
        + 1.25 * np.log(power_base)
        - np.log(1.0 + q)
        - L * instanton_action_density(gamma)
    )


def instanton_gap(L: int, gamma: float, prefactor: str = "garg"):
    """Return the published large-L WKB/instanton tunnelling gap.

    ``prefactor='garg'`` uses the result obtained from Garg's full WKB
    prefactor.  ``prefactor='isakov-printed'`` reproduces the expression as
    printed in the supplementary source of Isakov et al.
    """
    return float(np.exp(instanton_log_gap(L, gamma, prefactor)))


def estimate_uniform_steps(log_reference_gap: float, L: int):
    """Estimate Route 1 steps for the fixed Gamma=0.5 benchmark.

    The measured ``L=20`` and ``L=24`` rows give

        steps * gap / L = 57.25

    for the fixed magnetisation thresholds used by :func:`uniform_ite_gap`.
    The estimate must not be presented as a measured run or transferred to a
    different ``gamma`` without recalibration.
    """
    return float(np.log(57.25 * L) - log_reference_gap)


def uniform_ite_gap(
    L: int,
    gamma: float,
    upper_threshold: float = 1.0e-3,
    lower_threshold: float = 1.0e-6,
    check_every: int = 100,
    max_steps: int = 20_000_000,
):
    """Extract the doublet gap by uniform Crank--Nicolson evolution.

    The initial state is |m=S>, localised in one ferromagnetic well.  No Z2
    parity sector is selected.  A rigorous scalar lower-bound shift,

        H_shifted = H + L (Gamma + 1/2) I,

    makes the spectrum non-negative without using the exact ground energy.
    The time step obeys dtau * Hmax_bound = 1/2, so high-energy grid modes are
    damped by CN rather than left in its large-energy |r| -> 1 tail.

    Once |<S_z>|/S has fallen between the two small thresholds, higher states
    are negligible and its logarithmic decay determines r1/r0.  The exact CN
    amplification function is then inverted, using the late-state Rayleigh
    quotient for E0.  No reference eigenvalue enters this calculation.
    """
    if not 0.0 < lower_threshold < upper_threshold < 1.0:
        raise ValueError("require 0 < lower_threshold < upper_threshold < 1")
    if check_every <= 0:
        raise ValueError("check_every must be positive")

    diag, off, m = collective_hamiltonian(L, gamma)
    dimension = diag.size

    # ||-2 Gamma Sx|| <= Gamma L and ||-2 Sz^2/L|| <= L/2.
    shift = L * (gamma + 0.5)
    shifted_diag = diag + shift
    hmax_bound = 2.0 * shift
    dtau = 0.5 / hmax_bound
    coefficient = 0.5 * dtau

    left = np.zeros((3, dimension))
    left[0, 1:] = coefficient * off
    left[1, :] = 1.0 + coefficient * shifted_diag
    left[2, :-1] = coefficient * off
    right_diag = 1.0 - coefficient * shifted_diag
    right_off = -coefficient * off

    state = np.zeros(dimension)
    state[-1] = 1.0
    crossings = []
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        rhs = right_diag * state
        rhs[1:] += right_off * state[:-1]
        rhs[:-1] += right_off * state[1:]
        state = solve_banded((1, 1), left, rhs, check_finite=False)
        state /= np.linalg.norm(state)

        if step % check_every:
            continue

        magnetisation = abs(np.dot(state, m * state)) / (L / 2.0)
        target = upper_threshold if not crossings else lower_threshold
        if magnetisation > target:
            continue

        energy = float(np.dot(state, apply_tridiagonal(diag, off, state)))
        crossings.append((step, magnetisation, energy))
        if len(crossings) == 2:
            break
    else:
        raise RuntimeError(
            f"lower magnetisation threshold was not reached in {max_steps} steps"
        )

    (n_upper, m_upper, _), (n_lower, m_lower, late_energy) = crossings
    log_ratio_per_step = np.log(m_upper / m_lower) / (n_lower - n_upper)

    # CN factor r(E')=(1-aE')/(1+aE'), with a=dtau/2.  The late
    # Rayleigh quotient differs from E0 only quadratically in the already tiny
    # residual odd-state amplitude.
    ground_shifted = late_energy + shift
    r0 = (1.0 - coefficient * ground_shifted) / (
        1.0 + coefficient * ground_shifted
    )
    r1 = r0 * np.exp(-log_ratio_per_step)
    excited_shifted = (1.0 - r1) / (coefficient * (1.0 + r1))
    gap = excited_shifted - ground_shifted

    return ITEResult(
        gap=float(gap),
        steps=step,
        tau=float(step * dtau),
        wall_seconds=float(time.perf_counter() - started),
        upper_magnetisation=float(m_upper),
        lower_magnetisation=float(m_lower),
    )
