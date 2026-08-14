# Disaster-Response UAV--HAP Emergency Backhaul — Reference Implementation

Full, runnable Python implementation of the analytical + simulation
framework described in *"A Disaster-Response System Framework for
Age-Aware UAV--HAP Emergency Backhaul in 5G/6G Non-Terrestrial Networks."*

## Files

| File | Contents |
|---|---|
| `uav_hap_core.py` | All math: config, channel/service-time moments (Gauss–Laguerre, truncated), tandem priority M/G/1 delay model, PAoI, resilience metrics (CSPR/WSS/GDI/AWSS/AGDI/bottleneck indicator), discrete-event tandem-priority simulator, and Algorithm 1 (exhaustive-search orchestrator). |
| `plotting_utils.py` | Shared academic Matplotlib style (IEEE-like: grid, serif font, clean legends, 300-dpi PDF export). |
| `run_experiments.py` | Reproduces Experiments 1–6 of Sec. IX, prints all numerical tables to stdout, and writes every figure to `figures/`. |
| `figures/` | All generated `.pdf` figures. |

## Run

```bash
pip install numpy scipy matplotlib
python3 run_experiments.py
```

Everything (tables + 12 figures) is produced in one pass, in well under a
minute.

## Equation ↔ code map (most important ones)

| Paper Eq. | Meaning | Code location |
|---|---|---|
| (1) | aggregate offered rate λ_k | `Config.lam()` |
| (5) | admitted rate ã_k·λ_k | `evaluate_analytical`, `simulate_tandem` |
| (9) | service time S_k^(ij) = L_k/R_ij | `truncated_service_moments` (via c_{k,ij}) |
| (11) | ρ_k^X = λ̃_k^X·E[S_k^X] | `priority_stage_delays` |
| (14) | W_0^X (residual service term) | `priority_stage_delays` |
| (15) | E[W_i^X] non-preemptive priority M/G/1 waiting time | `priority_stage_delays` |
| (16)-(17) | E[T_i^X], tandem E2E delay | `priority_stage_delays`, `evaluate_analytical` |
| (18) | sample-path time-average AoI | `simulate_tandem` (`AoI_avg`) |
| (19)-(20) | peak age / mean PAoI | `simulate_tandem` (`PAoI_mean`), `evaluate_analytical` (`PAoI`) |
| (21)-(27) | outage-aware truncated service moments, Gauss–Laguerre quadrature | `truncated_service_moments` |
| (28)-(29) | LoS/NLoS mixture on HAP→GW hop | `service_moments_HG` |
| "opt_main" (Sec. VII-C) | joint orchestration objective + constraints | `objective_J`, `orchestrate` |
| Algorithm 1 | scenario-aware exhaustive search | `orchestrate` |
| Sec. VIII (CSPR/WSS/GDI/AWSS/AGDI/D_ρ) | resilience metrics | `cspr`, `wss_soft`, `gdi`, `awss_soft`, `agdi`, `bottleneck_indicator` |

## Notes on baseline parameters

The manuscript's numeric baseline table is generated at compile time via
`\input{tables/generated_baseline_table}`, which is not included in the
supplied `.tex` source — so the exact constants (packet sizes, offered
rates, bandwidth budget, SLA targets, orchestration weights) used to
produce the paper's *specific* numbers (e.g. "6.2% / 3.0% error", "103 s
PAoI explosion") are not recoverable from the LaTeX alone. This
implementation therefore uses a self-consistent, clearly documented
baseline (`Config` dataclass in `uav_hap_core.py`) chosen so that:

* the tandem analytical/simulation errors stay in the same few-percent
  ballpark reported in the paper (see Experiment 1 output),
* the P1 → P2 → P3 narrative of Experiment 5 reproduces qualitatively
  (P1 destabilizes first, P2 destabilizes next because it never throttles
  class 3, P3 remains feasible by admission-controlling class 3).

All constants are grouped in one place (`Config`) so you can drop in the
paper's real baseline table values directly if/when available.