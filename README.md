# Two Routes to Quantum Tunnelling — poster code

Minimal numerical code and recorded data accompanying Syed Hussain Haider's
2026 research poster, **Two Routes to Quantum Tunnelling: Numerical cost,
instanton accuracy and the role of symmetry**.

The poster authors are Syed Hussain Haider, Qiongyuan Wu and James Millen.
Contact: syedhussainhaider137@gmail.com.

This package contains only the calculations needed for the poster: the
original uniform-step imaginary-time calculation, the leading instanton
approximation and the uniform-step parity-resolved calculation. It does not
contain the full research workspace, correspondence or other investigations.

## Model and results

We use the fully connected transverse-field Ising (Curie–Weiss) Hamiltonian

    H = -2 Gamma Sx - (2/L) Sz^2,  S = L/2,  Gamma = 0.5,

with interaction strength and hbar set to one. L is an even number of spins.
Permutation symmetry reduces the representation to L+1 states. This is not a
nearest-neighbour Ising chain.

The original uniform Crank–Nicolson (CN) calculation does not separate parity
sectors. It extracts the tunnelling splitting from late-time magnetisation
decay and was run through L=24. The parity calculation instead evolves the
lowest state in each sector and reads the splitting using a discrete Wronskian
identity. Its target is 1e-10 **relative error**, not percent.

Recorded headline values:

- Original L=24: 12,224,900 steps, 86.7584 seconds; leading instanton error 3.7728%.
- Parity-resolved uniform CN at L=2800: 46,380 sector solves, 1.6187 seconds;
  leading instanton error 0.0282%.

These runs use different initial states, readouts, step choices and stopping
criteria. Their ratio is not a controlled measurement of a parity-only
speedup. Timings are historical observations and depend on the machine and
numerical libraries. No method is claimed to be the fastest endpoint solver.

## Files

| File | Purpose |
|---|---|
| `original.py` | Original uniform-CN solver, stationary reference, leading instanton formula and original step-count estimate |
| `parity.py` | Numerical functions needed for uniform parity CN and Wronskian readout |
| `reproduce.py` | Small runner for the recorded tables, calculation and runtime estimate |
| `data/original.csv` | Four measured original-method rows displayed in Table 1, with their measured errors |
| `data/parity.csv` | Three recorded parity rows displayed in Table 2 |
| `data/instanton.csv` | Seventeen points used in the instanton-error figure |
| `requirements.txt` | Numerical dependencies |

Numerical functions were extracted from the original research scripts without
changing their arithmetic. Unrelated experiments and command-line options were
excluded. Generic numerical helpers retain their original implementations;
the runner selects only the uniform-CN settings used here. The small runner
and documentation were prepared for this public release. It reproduces the
numerical evidence, not the poster's PowerPoint layout.

## Run

Requires Python 3.9 or later, NumPy and SciPy. From this directory:

```sh
python -m pip install -r requirements.txt
python reproduce.py recorded
python reproduce.py estimate
python reproduce.py run --quick
python reproduce.py run
```

`recorded` prints the original saved data. `run --quick` runs the original
solver at L=8, all three parity cases, and all instanton-accuracy points.
`run` additionally includes the slow original cases through L=24; allow a few
minutes. Commands write JSON to standard output and do not overwrite data.
For example, redirect a fresh run with `python reproduce.py run > fresh-run.json`.
New runtimes are not expected to equal the historical times.

Before publication, the full runner was checked with Python 3.9.6, NumPy 2.0.2
and SciPy 1.13.1. All four original step counts and all three parity solve
counts reproduced exactly. Original errors remained below 0.0002%; parity
errors met 1e-10 relative. All 17 instanton points agreed with the saved data
within 1e-7 percentage points. This check does not establish compatibility with
every later library version or an independent proof of the readout identity.

## Accuracy, references and extrapolation

- The leading semiclassical calculation uses the Garg-derived prefactor,
  proportional to `(1-Gamma^2)^(5/4)`. No fitted higher-order corrections enter
  the poster. The original formula helper retains the alternative printed
  Isakov prefactor for provenance; the runner always uses the Garg form.
- At small L, stationary eigenvalue subtraction provides the reference. At
  larger L, stationary parity eigenvectors and the Wronskian identity avoid
  cancellation. The parity benchmark and its reference share that readout;
  they use different state-preparation procedures.
- Parity results are the first checkpoints meeting the reference-graded error
  target. This is a reproducible benchmark, not a certified autonomous stopping
  rule. Exact reference vectors are not inserted into the evolution. The
  Gershgorin shift's roundoff margin uses the reference energy scale, as in the
  original benchmark.
- Original-method costs beyond L=24 are extrapolated, not measured. The model
  `N_steps ≈ 57.25 L / DeltaE` applies to the original Gamma=0.5 settings.
  Holding the measured L=24 seconds-per-step fixed gives about **1.3e17 seconds
  (4.2 billion years)** at L=100. This is an illustrative extrapolation, not a
  timing measurement, a rigorous lower bound, or a forecast for an optimised
  1%-accuracy solver. Per-step cost need not stay constant as L grows.
- A computational coverage gap depends on the accuracy and runtime budget.
  At a loose 5% target, the original data already overlap at L=20. The poster's
  schematic uses a stricter 1% target and a short runtime budget.

## Literature and citation

1. S. V. Isakov et al., *Understanding quantum tunneling through quantum Monte
   Carlo simulations*, Phys. Rev. Lett. **117**, 180402 (2016).
2. A. Garg, J. Math. Phys. **39**, 5166 (1998),
   [doi:10.1063/1.532563](https://doi.org/10.1063/1.532563).
3. C. Herring, Rev. Mod. Phys. **34**, 631 (1962).

Suggested code citation: S. H. Haider, *Two Routes to Quantum Tunnelling —
poster code* (2026). Cite the repository URL and the commit used so the
numerical implementation is identifiable.
