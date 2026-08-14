# diagnostic_variance_check.py
import numpy as np
import pandas as pd
from uav_hap_core import Config, evaluate_analytical
from run_experiments import _policy_candidate, _simulate_multiseed, CFG, CLASSES

DIAG_SEEDS = tuple(202601 + i for i in range(200))  # đủ để thấy xu hướng std, chưa cần full 1000
SIM_HORIZON, SIM_WARMUP = 7000.0, 1000.0

# Vùng A: quanh đảo dấu thật (a3: 0.25 -> 0.0), kappa 7.4-7.8
# Vùng B: đối chứng, throttle liên tục không đảo dấu (a2: 1.0->0.75), kappa 10.3-10.7
kappa_probe = [7.4, 7.5, 7.6, 7.7, 7.8, 10.3, 10.4, 10.5, 10.6, 10.7]

rows = []
for kappa in kappa_probe:
    res = _policy_candidate(CFG, "P3", kappa)
    if res is None:
        print(f"kappa={kappa}: INFEASIBLE"); continue
    sim = _simulate_multiseed(CFG, res["alpha"], res["B_UH"], res["B_HG"],
                               res["a"], kappa=kappa,
                               horizon=SIM_HORIZON, warmup=SIM_WARMUP,
                               seeds=DIAG_SEEDS)
    k1 = sim[1]
    gap = res["PAoI"][1] - k1["AoI_avg"]
    rows.append(dict(
        kappa=kappa, a3=res["a"][3], a2=res["a"][2],
        PAoI1_ana=res["PAoI"][1], AoI1_sim_mean=k1["AoI_avg"],
        gap_sim_paoi=gap,
        E2E_cv=k1["E2E_cv_across_seeds"], AoI_cv=k1["AoI_cv_across_seeds"],
        E2E_p95=k1["E2E_p95"], AoI_p95=k1["AoI_p95"],
        n_seeds=k1["n_seeds_valid"],
    ))
    print(f"kappa={kappa:.1f}  a3={res['a'][3]:.2f}  gap={gap:+.4f}  "
          f"E2E_cv={k1['E2E_cv_across_seeds']:.4f}  AoI_cv={k1['AoI_cv_across_seeds']:.4f}  "
          f"AoI_p95={k1['AoI_p95']:.4f}")

df = pd.DataFrame(rows)
df.to_csv("csv_tables/diag_variance_check_kappa_transitions.csv", index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))