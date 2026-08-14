import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor, as_completed

from uav_hap_core import (Config, evaluate_analytical, simulate_tandem,
                           orchestrate, objective_J, cspr, wss_soft, gdi,
                           awss_soft, agdi, bottleneck_indicator,
                           outage_probability)
# from validation_figure import FIG_DIR

CSV_DIR = os.path.join(os.getcwd(), "csv_tables")
os.makedirs(CSV_DIR, exist_ok=True)

CFG = Config()
CLASSES = CFG.classes

SIM_HORIZON = 7000.0
SIM_WARMUP = 1000.0
SEED_LIST = tuple(202601 + i for i in range(1000))

ALPHA_GRID_ORCHESTRATION = [0.35, 0.45, 0.55, 0.65]
BUH_GRID_ORCHESTRATION = [0.6e6, 0.8e6, 1.0e6, 1.2e6, 1.4e6]
ALPHA_MIN_ORCHESTRATION = min(ALPHA_GRID_ORCHESTRATION)

N_SWEEP = 200
SWEEP_MARGIN_FRAC = 0.01
ALPHA_SWEEP_DENSE = np.linspace(0.05, 0.95, N_SWEEP)
BUH_SWEEP_DENSE = np.linspace(SWEEP_MARGIN_FRAC * CFG.B_tot,
                               (1 - SWEEP_MARGIN_FRAC) * CFG.B_tot,
                               N_SWEEP)

S2_BAR_GAMMA_UH = 4.0
N_WORKERS = int(os.environ.get("N_WORKERS", os.cpu_count() or 4))


def _finite_or_nan(x):
    try:
        return float(x) if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _write_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[saved csv] {path}  ({len(df)} rows)")


# ==============================================================================
# PARALLEL SEED WORKER
# ==============================================================================
def _one_seed_worker(args):
    """Runs simulate_tandem() for a single seed. Must be a top-level
    (module-level) function, not a closure/lambda, so it can be pickled
    and sent to worker processes by ProcessPoolExecutor."""
    (cfg, alpha, B_UH, B_HG, a, kappa, horizon, warmup, seed,
     channel_kwargs) = args
    sim = simulate_tandem(cfg, alpha, B_UH, B_HG, a, kappa=kappa,
                           horizon=horizon, warmup=warmup, seed=seed,
                           channel_kwargs=channel_kwargs)
    return sim

_POOL = None


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=N_WORKERS)
    return _POOL


def shutdown_pool():
    """Call this once at the very end of the script (see __main__ below)."""
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None


def _simulate_multiseed(cfg, alpha, B_UH, B_HG, a, kappa=1.0,
                         horizon=SIM_HORIZON, warmup=SIM_WARMUP,
                         seeds=SEED_LIST, channel_kwargs=None):
    channel_kwargs = channel_kwargs or {}
    pool = _get_pool()

    tasks = [(cfg, alpha, B_UH, B_HG, a, kappa, horizon, warmup, sd,
              channel_kwargs) for sd in seeds]

    per_class_runs = {k: {"E2E_mean": [], "PAoI_mean": [], "AoI_avg": []}
                       for k in CLASSES}
    rho_U_sim_runs, rho_H_sim_runs = [], []

    chunksize = max(1, len(tasks) // (N_WORKERS * 4))
    for sim in pool.map(_one_seed_worker, tasks, chunksize=chunksize):
        for k in CLASSES:
            for field in ("E2E_mean", "PAoI_mean", "AoI_avg"):
                v = sim[k][field]
                if np.isfinite(v):
                    per_class_runs[k][field].append(v)
        if np.isfinite(sim.get('_rho_U_sim', np.nan)):
            rho_U_sim_runs.append(sim['_rho_U_sim'])
        if np.isfinite(sim.get('_rho_H_sim', np.nan)):
            rho_H_sim_runs.append(sim['_rho_H_sim'])

    out = {}
    for k in CLASSES:
        vals_e2e = per_class_runs[k]["E2E_mean"]
        vals_aoi = per_class_runs[k]["AoI_avg"]
        out[k] = {field: (float(np.mean(vals)) if vals else np.nan)
                   for field, vals in per_class_runs[k].items()}
        out[k]["n_seeds_valid"] = len(vals_e2e)
        out[k]["E2E_std_across_seeds"] = float(np.std(vals_e2e, ddof=1)) if len(vals_e2e) > 1 else np.nan
        out[k]["AoI_std_across_seeds"] = float(np.std(vals_aoi, ddof=1)) if len(vals_aoi) > 1 else np.nan
        out[k]["E2E_p05"] = float(np.percentile(vals_e2e, 5)) if vals_e2e else np.nan
        out[k]["E2E_p95"] = float(np.percentile(vals_e2e, 95)) if vals_e2e else np.nan
        out[k]["AoI_p05"] = float(np.percentile(vals_aoi, 5)) if vals_aoi else np.nan
        out[k]["AoI_p95"] = float(np.percentile(vals_aoi, 95)) if vals_aoi else np.nan
        out[k]["E2E_cv_across_seeds"] = (out[k]["E2E_std_across_seeds"] / out[k]["E2E_mean"]
                                          if vals_e2e and out[k]["E2E_mean"] else np.nan)
        out[k]["AoI_cv_across_seeds"] = (out[k]["AoI_std_across_seeds"] / out[k]["AoI_avg"]
                                          if vals_aoi and out[k]["AoI_avg"] else np.nan)
    out['_rho_U_sim'] = float(np.mean(rho_U_sim_runs)) if rho_U_sim_runs else np.nan
    out['_rho_H_sim'] = float(np.mean(rho_H_sim_runs)) if rho_H_sim_runs else np.nan
    return out



def experiment1_seed_convergence_check_all_scenarios(
        seed_counts=tuple(range(200, 400, 10))):
    scenarios = {
        "S4 (disaster surge, kappa=1.6)": dict(kappa=1.6, overrides={}),
    }
    a = {1: 1.0, 2: 1.0, 3: 1.0}
    pool = tuple(202601 + i for i in range(max(seed_counts)))

    all_rows = []
    for sname, sk in scenarios.items():
        kappa, overrides = sk["kappa"], sk["overrides"]
        ana = evaluate_analytical(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=kappa, **overrides)
        for n in seed_counts:
            seeds = pool[:n]
            sim = _simulate_multiseed(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                       kappa=kappa, horizon=SIM_HORIZON,
                                       warmup=SIM_WARMUP, seeds=seeds,
                                       channel_kwargs=overrides)
            re_list = []
            for k in CLASSES:
                T_ana = _finite_or_nan(ana["T_E2E"][k])
                T_sim = _finite_or_nan(sim[k]["E2E_mean"])
                if T_sim:
                    re_list.append(abs(T_ana - T_sim) / T_sim * 100)
            all_rows.append(dict(scenario=sname, n_seeds=n,
                                  RE_delay_mean_pct=np.mean(re_list),
                                  RE_delay_max_pct=np.max(re_list)))
        print(f"[{sname}] done.")

    df = pd.DataFrame(all_rows)
    df["rolling_mean_last3"] = (df.groupby("scenario")["RE_delay_mean_pct"]
                                 .transform(lambda s: s.rolling(3).mean()))
    _write_csv(df, "exp1_seed_convergence_check_all_scenarios.csv")
    return df


# ==============================================================================
# EXPERIMENT 1 -- Validation of the tandem approximation
# ==============================================================================
def experiment1_validation():
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: Validation of the tandem M/G/1 priority approximation")
    print("=" * 78)

    scenarios = {
        "S1 (nominal, kappa=1.0)": dict(kappa=1.0, overrides={}),
        "S2 (weak UAV-HAP link, kappa=1.0)": dict(
            kappa=1.0, overrides=dict(bar_gamma_UH=S2_BAR_GAMMA_UH)),
        "S3 (degraded backhaul, kappa=1.0)": dict(
            kappa=1.0, overrides=dict(p_LoS=0.3,
                                       bar_gamma_HG_LoS=7.5,
                                       bar_gamma_HG_NLoS=2.5)),
        "S4 (disaster surge, kappa=1.6)": dict(kappa=1.6, overrides={}),
    }

    detail_rows = []
    summary_rows = []
    for sname, sk in scenarios.items():
        kappa = sk["kappa"]
        overrides = sk["overrides"]
        a = {1: 1.0, 2: 1.0, 3: 1.0}

        ana = evaluate_analytical(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=kappa, **overrides)

        sim = _simulate_multiseed(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=kappa, horizon=SIM_HORIZON,
                                   warmup=SIM_WARMUP, seeds=SEED_LIST,
                                   channel_kwargs=overrides)

        re_delay_list, re_paoi_list = [], []
        for k in CLASSES:
            T_ana = _finite_or_nan(ana["T_E2E"][k])
            T_sim = _finite_or_nan(sim[k]["E2E_mean"])
            re = abs(T_ana - T_sim) / T_sim * 100 if T_sim else np.nan
            A_ana = _finite_or_nan(ana["PAoI"][k])
            A_sim = _finite_or_nan(sim[k]["PAoI_mean"])
            re_a = abs(A_ana - A_sim) / A_sim * 100 if A_sim else np.nan
            detail_rows.append(dict(
                scenario=sname, kappa=kappa, cls=k,
                T_ana_s=T_ana, T_sim_s=T_sim, RE_delay_pct=re,
                PAoI_ana_s=A_ana, PAoI_sim_s=A_sim, RE_paoi_pct=re_a,
                rho_U=ana["rho_U_tot"], rho_H=ana["rho_H_tot"],
                n_seeds_valid=sim[k]["n_seeds_valid"]))
            if np.isfinite(re):
                re_delay_list.append(re)
            if np.isfinite(re_a):
                re_paoi_list.append(re_a)

        summary_rows.append(dict(
            scenario=sname, kappa=kappa,
            rho_U_ana=ana["rho_U_tot"], rho_U_sim=sim.get('_rho_U_sim', np.nan),
            rho_H_ana=ana["rho_H_tot"], rho_H_sim=sim.get('_rho_H_sim', np.nan),
            RE_delay_mean_pct=np.mean(re_delay_list) if re_delay_list else np.nan,
            RE_paoi_mean_pct=np.mean(re_paoi_list) if re_paoi_list else np.nan,
        ))

    df_detail = pd.DataFrame(detail_rows)
    df_summary = pd.DataFrame(summary_rows)

    print("\n-- Per-scenario summary (mean/max across the 3 classes) --")
    print(df_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    global_re_delay = df_detail["RE_delay_pct"].dropna()
    global_re_paoi = df_detail["RE_paoi_pct"].dropna()
    print(f"\n[GLOBAL] RE_delay mean={global_re_delay.mean():.4f}%  "
          f"max={global_re_delay.max():.4f}%")
    print(f"[GLOBAL] RE_paoi  mean={global_re_paoi.mean():.4f}%  "
          f"max={global_re_paoi.max():.4f}%")

    _write_csv(df_detail, "exp1_validation_delay_paoi_detail.csv")
    _write_csv(df_summary, "exp1_validation_delay_paoi_summary.csv")
    return df_detail, df_summary


# ==============================================================================
# EXPERIMENT 2 -- UAV--HAP SNR sweep and bottleneck migration
# ==============================================================================
def experiment2_snr_sweep():
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Effect of UAV--HAP SNR (bar_gamma_UH) on delay / bottleneck")
    print("=" * 78)
    snr_list = np.linspace(2, 100.0, 99)
    a = {1: 1.0, 2: 1.0, 3: 1.0}

    rows = []
    for g in snr_list:
        res = evaluate_analytical(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=1.0, bar_gamma_UH=g)
        Drho = bottleneck_indicator(res)
        rows.append(dict(
            bar_gamma_UH=g,
            T1_E2E_s=_finite_or_nan(res["T_E2E"][1]),
            T2_E2E_s=_finite_or_nan(res["T_E2E"][2]),
            T3_E2E_s=_finite_or_nan(res["T_E2E"][3]),
            rho_U=res["rho_U_tot"], rho_H=res["rho_H_tot"], D_rho=Drho,
        ))
        print(f"bar_gamma_UH={g:5.1f}  rho_U={res['rho_U_tot']:.3f}  "
              f"rho_H={res['rho_H_tot']:.3f}  D_rho={Drho:+.3f}  "
              f"T1={res['T_E2E'][1]:.4f}s  T3={res['T_E2E'][3]:.4f}s")

    df = pd.DataFrame(rows)
    _write_csv(df, "exp2_snr_sweep_delay_bottleneck.csv")
    return df


# ==============================================================================
# EXPERIMENT 3 -- Half-duplex split (alpha) sweep [DENSE, continuous-style]
# ==============================================================================
def experiment3_alpha_sweep():
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: Effect of the half-duplex split alpha (weak link S2)")
    print("=" * 78)
    a = {1: 1.0, 2: 1.0, 3: 1.0}
    weak_gamma_UH = S2_BAR_GAMMA_UH

    rows = []
    for alpha in ALPHA_SWEEP_DENSE:
        res = evaluate_analytical(CFG, alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=1.0, bar_gamma_UH=weak_gamma_UH)
        W_U = {k: res["T_U"][k] - res["mom_U"][k][0] for k in CLASSES}
        rows.append(dict(
            alpha=alpha,
            W1_U_s=_finite_or_nan(W_U[1]), W2_U_s=_finite_or_nan(W_U[2]),
            W3_U_s=_finite_or_nan(W_U[3]),
            T1_E2E_s=_finite_or_nan(res["T_E2E"][1]),
            T2_E2E_s=_finite_or_nan(res["T_E2E"][2]),
            T3_E2E_s=_finite_or_nan(res["T_E2E"][3]),
            rho_U=res["rho_U_tot"], rho_H=res["rho_H_tot"],
        ))
        print(f"alpha={alpha:.4f}  rho_U={res['rho_U_tot']:.4f}  "
              f"W1_U={W_U[1]:.4f}s  T1_E2E={res['T_E2E'][1]:.4f}s")

    df = pd.DataFrame(rows)

    # --- Stability boundary (interpolated), same method as Exp.2's SNR
    # crossover. --------------------------------------------------------
    last_stable_idx = df["T1_E2E_s"].last_valid_index()
    first_unstable_idx = (last_stable_idx + 1
                           if last_stable_idx is not None
                           and last_stable_idx + 1 < len(df) else None)
    if first_unstable_idx is not None:
        alpha_lo = df.loc[last_stable_idx, "alpha"]
        alpha_hi = df.loc[first_unstable_idx, "alpha"]
        rho_lo = df.loc[last_stable_idx, "rho_U"]
        rho_hi = df.loc[first_unstable_idx, "rho_U"]
        alpha_stab = alpha_lo + (1.0 - rho_lo) / (rho_hi - rho_lo) * (alpha_hi - alpha_lo)
        print(f"\n[STABILITY BOUNDARY] alpha_stab ~ {alpha_stab:.4f} "
              f"(interpolated between alpha={alpha_lo:.4f}, rho_U={rho_lo:.4f} "
              f"and alpha={alpha_hi:.4f}, rho_U={rho_hi:.4f})")
        df.attrs["alpha_stab"] = alpha_stab
    else:
        print("\n[STABILITY BOUNDARY] not reached within ALPHA_SWEEP_DENSE.")
        df.attrs["alpha_stab"] = np.nan
        last_stable_idx = df.index[-1]

    # Reference chord between the first and last stable points (convexity
    # check for rho_U(alpha), discussed alongside Fig. rho_vs_alpha).
    first_idx = df.index[0]
    df["rho_U_reference_line"] = np.interp(
        df["alpha"],
        [df.loc[first_idx, "alpha"], df.loc[last_stable_idx, "alpha"]],
        [df.loc[first_idx, "rho_U"], df.loc[last_stable_idx, "rho_U"]])

    # --- Quantitative amplification factor near the stability boundary vs.
    # near the start of the sweep (Patch [10]). Turns the qualitative
    # LaTeX claim ("growth is far steeper... near the boundary") into a
    # citable number, using local slopes over the first/last ~10% of the
    # stable alpha-range. --------------------------------------------------
    # Filter to physically stable region only (rho_U < 1), not just finite values
    stable = df[df["rho_U"] < 1.0].dropna(subset=["T1_E2E_s"]).reset_index(drop=True)
    amp_factor = np.nan
    if len(stable) >= 10:
        n = len(stable)
        span = max(2, n // 10)
        head, tail = stable.iloc[:span], stable.iloc[-span:]

        def _local_slope(block):
            da = block["alpha"].iloc[-1] - block["alpha"].iloc[0]
            dT = block["T1_E2E_s"].iloc[-1] - block["T1_E2E_s"].iloc[0]
            return dT / da if da != 0 else np.nan

        slope_head, slope_tail = _local_slope(head), _local_slope(tail)
        if slope_head and np.isfinite(slope_head) and slope_head != 0:
            amp_factor = slope_tail / slope_head
        print(f"[AMPLIFICATION] local dT1/dalpha near start = {slope_head:.4f} s/unit, "
              f"near stability boundary = {slope_tail:.4f} s/unit, "
              f"amplification factor = {amp_factor:.2f}x")
    df.attrs["amplification_factor"] = amp_factor

    _write_csv(df, "exp3_alpha_sweep_waiting_delay.csv")
    return df


# ==============================================================================
# EXPERIMENT 4 -- Bandwidth partition and bottleneck balancing [DENSE]
# ==============================================================================
def _find_true_optimum_B_UH(cfg, a, w_delay, margin_frac=0.01):
    B_UH_min = margin_frac * cfg.B_tot
    B_UH_max = (1 - margin_frac) * cfg.B_tot

    def objective(B_UH):
        B_HG = cfg.B_tot - B_UH
        res = evaluate_analytical(cfg, cfg.alpha, B_UH, B_HG, a, kappa=1.0)
        if res["rho_U_tot"] >= 1.0 or res["rho_H_tot"] >= 1.0:
            return 1e12  # penalize infeasible region
        return sum(w_delay[k] * res["T_E2E"][k] for k in CLASSES)

    result = minimize_scalar(objective, bounds=(B_UH_min, B_UH_max),
                              method='bounded',
                              options={'xatol': 1.0})  # tolerance in Hz
    return result.x, result.fun


def experiment4_bandwidth_split():
    print("\n" + "=" * 78)
    print("EXPERIMENT 4: Bandwidth partition B_UH vs weighted delay objective P")
    print("=" * 78)
    a = {1: 1.0, 2: 1.0, 3: 1.0}

    rows = []
    for B_UH in BUH_SWEEP_DENSE:
        B_HG = CFG.B_tot - B_UH
        res = evaluate_analytical(CFG, CFG.alpha, B_UH, B_HG, a, kappa=1.0)
        if res["rho_U_tot"] >= 1.0 or res["rho_H_tot"] >= 1.0:
            continue
        P = sum(CFG.w_delay[k] * res["T_E2E"][k] for k in CLASSES)
        rows.append(dict(
            B_UH_MHz=B_UH / 1e6, B_HG_MHz=B_HG / 1e6, P_weighted_delay=P,
            rho_U=res["rho_U_tot"], rho_H=res["rho_H_tot"],
            rho_gap_abs=abs(res["rho_U_tot"] - res["rho_H_tot"]),
            T1_E2E_s=res["T_E2E"][1], T2_E2E_s=res["T_E2E"][2],
            T3_E2E_s=res["T_E2E"][3],
        ))

    df = pd.DataFrame(rows)
    grid_best_idx = int(df["P_weighted_delay"].idxmin())
    df["is_grid_dense_optimal"] = False
    df.loc[grid_best_idx, "is_grid_dense_optimal"] = True

    print(f"[DENSE GRID N={N_SWEEP}] argmin P={df.loc[grid_best_idx,'P_weighted_delay']:.4f} "
          f"at B_UH={df.loc[grid_best_idx,'B_UH_MHz']:.3f} MHz "
          f"(rho_U={df.loc[grid_best_idx,'rho_U']:.3f}, "
          f"rho_H={df.loc[grid_best_idx,'rho_H']:.3f}, "
          f"|gap|={df.loc[grid_best_idx,'rho_gap_abs']:.3f}) "
          f"[B_tot={CFG.B_tot/1e6:.1f} MHz]")

    # --- (a) True continuous optimum: sole source of truth for B_UH*, P*
    # quoted anywhere in the paper text (never taken from the dense-grid
    # argmin above, to avoid quoting two slightly different numbers for
    # the same quantity). ----------
    B_UH_star, P_star = _find_true_optimum_B_UH(CFG, a, CFG.w_delay,
                                                  margin_frac=0.01)
    B_HG_star = CFG.B_tot - B_UH_star
    res_star = evaluate_analytical(CFG, CFG.alpha, B_UH_star, B_HG_star, a, kappa=1.0)
    rho_gap_star = abs(res_star["rho_U_tot"] - res_star["rho_H_tot"])
    print(f"[TRUE OPTIMUM, scipy bounded] B_UH*={B_UH_star/1e6:.4f} MHz  "
          f"P*={P_star:.4f}  rho_U={res_star['rho_U_tot']:.4f}  "
          f"rho_H={res_star['rho_H_tot']:.4f}  |gap|={rho_gap_star:.4f}")

    # --- (b) Balance-at-optimum vs. balance-at-edges, quantifying the
    # qualitative "optimum is near where utilizations are balanced" claim.
    edge_lo, edge_hi = df.iloc[0], df.iloc[-1]
    print(f"[BALANCE CHECK] |rho_U-rho_H| at true optimum = {rho_gap_star:.4f}  "
          f"vs. at low-B_UH edge ({edge_lo['B_UH_MHz']:.2f} MHz) = "
          f"{edge_lo['rho_gap_abs']:.4f}  vs. at high-B_UH edge "
          f"({edge_hi['B_UH_MHz']:.2f} MHz) = {edge_hi['rho_gap_abs']:.4f}")

    # --- (c) Performance loss of the coarse ORCHESTRATION grid (the one
    # Algorithm 1 actually uses in Exp.5/6/7) relative to the true
    # continuous optimum. Quantifies whether the "computationally light"
    # grid search of Sec. VII leaves meaningful performance on the table.
    orch_rows = []
    for B_UH in BUH_GRID_ORCHESTRATION:
        B_HG = CFG.B_tot - B_UH
        if B_HG <= 0:
            continue
        res = evaluate_analytical(CFG, CFG.alpha, B_UH, B_HG, a, kappa=1.0)
        if res["rho_U_tot"] >= 1.0 or res["rho_H_tot"] >= 1.0:
            continue
        P = sum(CFG.w_delay[k] * res["T_E2E"][k] for k in CLASSES)
        orch_rows.append(dict(B_UH_MHz=B_UH / 1e6, P_weighted_delay=P))
    df_orch = pd.DataFrame(orch_rows)
    orch_best_idx = int(df_orch["P_weighted_delay"].idxmin())
    P_orch_best = df_orch.loc[orch_best_idx, "P_weighted_delay"]
    B_UH_orch_best = df_orch.loc[orch_best_idx, "B_UH_MHz"]
    pct_loss = (P_orch_best - P_star) / P_star * 100.0
    print(f"[GRID-SEARCH GAP] coarse ORCHESTRATION grid picks "
          f"B_UH={B_UH_orch_best:.2f} MHz -> P={P_orch_best:.4f}  "
          f"({pct_loss:.3f}% above the true continuous optimum P*={P_star:.4f})")

    df.attrs["B_UH_star_MHz"] = B_UH_star / 1e6
    df.attrs["P_star"] = P_star
    df.attrs["rho_gap_star"] = rho_gap_star
    df.attrs["orchestration_grid_pct_loss"] = pct_loss
    df.attrs["orchestration_grid_B_UH_MHz"] = B_UH_orch_best

    _write_csv(df, "exp4_bandwidth_split_objective.csv")
    return df


def _plot_bandwidth_split_objective(cfg, a, w_delay, out_path,
                                     n_points=2000, margin_frac=0.01):
    """
    Near-continuous evaluation of P(B_UH) purely for plotting (kept
    separate from the CSV-producing sweep above, which already uses
    N_SWEEP=200 -- this finer grid is only for a smooth figure, not for
    any numeric claim in the text; all such claims come from
    _find_true_optimum_B_UH / experiment4_bandwidth_split()).
    """
    import matplotlib.pyplot as plt

    B_UH_min = margin_frac * cfg.B_tot
    B_UH_max = (1 - margin_frac) * cfg.B_tot
    B_UH_dense = np.linspace(B_UH_min, B_UH_max, n_points)

    P_vals = np.empty(n_points)
    rho_U_vals = np.empty(n_points)
    rho_H_vals = np.empty(n_points)
    for i, B_UH in enumerate(B_UH_dense):
        B_HG = cfg.B_tot - B_UH
        res = evaluate_analytical(cfg, cfg.alpha, B_UH, B_HG, a, kappa=1.0)
        P_vals[i] = sum(w_delay[k] * res["T_E2E"][k] for k in CLASSES)
        rho_U_vals[i] = res["rho_U_tot"]
        rho_H_vals[i] = res["rho_H_tot"]

    feasible = np.isfinite(P_vals) & (rho_U_vals < 1.0) & (rho_H_vals < 1.0)
    B_plot = B_UH_dense[feasible]
    P_plot = P_vals[feasible]
    rho_U_plot = rho_U_vals[feasible]
    rho_H_plot = rho_H_vals[feasible]

    best_i = int(np.argmin(P_plot))

    fig, ax1 = plt.subplots(figsize=(6.5, 4.8))
    ax1.plot(B_plot / 1e6, P_plot, '-', color='k', linewidth=1.4, label='P (weighted delay)')
    ax1.plot(B_plot[best_i] / 1e6, P_plot[best_i], marker='o', markersize=8,
              markerfacecolor='white', markeredgecolor='k', linestyle='None',
              label=f'Minimum ($B_{{UH}}={B_plot[best_i]/1e6:.3f}$ MHz)')
    ax1.set_xlabel(r'$B_{UH}$ [MHz]')
    ax1.set_ylabel(r'Weighted end-to-end delay $P$')

    # Overlay stage utilizations on a secondary axis so the reader can see
    # the crossover of rho_U/rho_H against the bottom of the U-shape.
    ax2 = ax1.twinx()
    ax2.plot(B_plot / 1e6, rho_U_plot, '--', color='tab:blue', linewidth=1.0,
              label=r'$\rho^U$')
    ax2.plot(B_plot / 1e6, rho_H_plot, '--', color='tab:red', linewidth=1.0,
              label=r'$\rho^H$')
    ax2.set_ylabel(r'Stage utilization $\rho$')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved fig] {out_path}  (dense B_UH sweep with rho overlay, "
          f"minimum P={P_plot[best_i]:.4f} at B_UH={B_plot[best_i]/1e6:.3f} MHz)")
    return B_plot, P_plot


# ==============================================================================
# EXPERIMENT 5 -- Priority-only vs resource-adaptive vs proposed orchestration
# ==============================================================================
def _policy_candidate(cfg, policy, kappa):
    if policy == "P1":
        a = {1: 1.0, 2: 1.0, 3: 1.0}
        B_eq = cfg.B_tot / 2.0
        res = evaluate_analytical(cfg, 0.5, B_eq, B_eq, a, kappa=kappa)
        if res["rho_U_tot"] >= 1.0 or res["rho_H_tot"] >= 1.0:
            return None
        J, U, P = objective_J(cfg, res, kappa=kappa)
        res = dict(res); res.update(J=J, U=U, P=P)
        return res
    elif policy == "P2":
        return orchestrate(cfg, kappa=kappa,
                            alpha_grid=ALPHA_GRID_ORCHESTRATION,
                            BUH_grid=BUH_GRID_ORCHESTRATION,
                            a1_grid=[1.0], a2_grid=[1.0], a3_grid=[1.0])
    elif policy == "P3":
        return orchestrate(cfg, kappa=kappa,
                            alpha_grid=ALPHA_GRID_ORCHESTRATION,
                            BUH_grid=BUH_GRID_ORCHESTRATION,
                            a1_grid=[1.0], a2_grid=[0.5, 0.75, 1.0],
                            a3_grid=[0.0, 0.25, 0.5, 0.75, 1.0])
    raise ValueError(policy)


def experiment5_policy_comparison():
    print("\n" + "=" * 78)
    print("EXPERIMENT 5: P1 (fixed) vs P2 (resource-adaptive) vs P3 (proposed)")
    print("=" * 78)
    kappa_list = [
        1.0, 1.5, 2.0, 2.5,
        2.6, 2.7, 2.8, 2.9, 3.0,
        3.2, 3.4, 3.6, 3.7, 3.8, 3.9,
        4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
        4.6, 4.7, 4.8,
        5.0, 5.5, 6.0, 6.5, 7.0,
        7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
        8.5, 9.0, 9.5, 10.0,
        10.1, 10.2, 10.3, 10.4, 10.5,
        11.0,
    ]
    policies = ["P1", "P2", "P3"]
    nominal = {p: _policy_candidate(CFG, p, 1.0) for p in policies}

    rows = []
    t0 = time.time()
    n_total = len(kappa_list) * len(policies)
    n_done = 0
    for kappa in kappa_list:
        for p in policies:
            res = _policy_candidate(CFG, p, kappa)
            n_done += 1
            if res is None:
                rows.append(dict(kappa=kappa, policy=p, feasible=False,
                                  alpha=np.nan, B_UH_MHz=np.nan,
                                  a1=np.nan, a2=np.nan, a3=np.nan,
                                  T1_E2E_s=np.nan, T2_E2E_s=np.nan, T3_E2E_s=np.nan,
                                  rho_U=np.nan, rho_H=np.nan,
                                  CSPR=np.nan, GDI=np.nan, AGDI=np.nan,
                                  PAoI1_s=np.nan, PAoI2_s=np.nan, PAoI3_s=np.nan,
                                  AoI_avg1_sim_s=np.nan, n_seeds_valid=0))
                print(f"[{p}] kappa={kappa:.1f}: INFEASIBLE  "
                      f"({n_done}/{n_total}, {time.time()-t0:.1f}s elapsed)")
                continue
            CSPR_v = cspr(CFG, res, kappa=kappa)
            GDI_v = gdi(CFG, res, nominal[p]) if nominal[p] else np.nan
            AGDI_v = agdi(CFG, res, nominal[p]) if nominal[p] else np.nan

            sim = _simulate_multiseed(CFG, res["alpha"], res["B_UH"], res["B_HG"],
                                       res["a"], kappa=kappa,
                                       horizon=SIM_HORIZON, warmup=SIM_WARMUP,
                                       seeds=SEED_LIST)
            AoI_avg1 = sim[1]["AoI_avg"]
            rows.append(dict(
                kappa=kappa, policy=p, feasible=True,
                alpha=res["alpha"], B_UH_MHz=res["B_UH"] / 1e6,
                a1=res["a"][1], a2=res["a"][2], a3=res["a"][3],
                T1_E2E_s=res["T_E2E"][1], T2_E2E_s=res["T_E2E"][2], T3_E2E_s=res["T_E2E"][3],
                rho_U=res["rho_U_tot"], rho_H=res["rho_H_tot"],
                CSPR=CSPR_v, GDI=GDI_v, AGDI=AGDI_v,
                PAoI1_s=res["PAoI"][1], PAoI2_s=res["PAoI"][2], PAoI3_s=res["PAoI"][3],
                AoI_avg1_sim_s=AoI_avg1, n_seeds_valid=sim[1]["n_seeds_valid"],
            ))
            print(f"[{p}] kappa={kappa:.1f}: T1={res['T_E2E'][1]:.4f}s  "
                  f"CSPR={CSPR_v:.3f}  GDI={GDI_v:.3f}  a={res['a']}  "
                  f"PAoI1={res['PAoI'][1]:.3f}s  simAoI1={AoI_avg1:.3f}s  "
                  f"({n_done}/{n_total}, {time.time()-t0:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    _write_csv(df, "exp5_policy_comparison_vs_kappa.csv")
    print(f"\n[TIMING] Experiment 5 total wall time: {time.time()-t0:.1f}s "
          f"with N_WORKERS={N_WORKERS}")
    return df


# ==============================================================================
# EXPERIMENT 6 -- Backhaul degradation (S3) and combined-stress (S5)
#                survivability  [DEBUG + RE-PARAMETERIZED]
# ==============================================================================

_AWSS_P1_DEBUG_STATE = {"printed_ok": False, "printed_bad": False, "printed_exc": False}


def _p1_baseline_exp6(cfg, kappa, channel_kwargs):
    """P1: fixed alpha=0.5, B_UH=B_HG=B_tot/2, full admission a=(1,1,1).
    Matches Sec. XI-B 'Compared Policies' definition of P1 exactly, and is
    evaluated here under the SAME channel_kwargs (p_LoS, etc.) as the P3
    candidate below, so the AWSS_soft(P1) vs AWSS_soft(P3) comparison is
    apples-to-apples at every (p_LoS, kappa) point of Exp.6."""
    a = {1: 1.0, 2: 1.0, 3: 1.0}
    B_eq = cfg.B_tot / 2.0
    res = evaluate_analytical(cfg, 0.5, B_eq, B_eq, a, kappa=kappa, **channel_kwargs)
    if res["rho_U_tot"] >= 1.0 or res["rho_H_tot"] >= 1.0:
        return None
    return res


def _p3_orchestrated_exp6(cfg, kappa, channel_kwargs):
    """P3: full Algorithm 1 search, exactly as used elsewhere in Exp.5/6/7
    (same candidate grids, same monotone-admission enforcement)."""
    return orchestrate(cfg, kappa=kappa,
                       alpha_grid=ALPHA_GRID_ORCHESTRATION,
                       BUH_grid=BUH_GRID_ORCHESTRATION,
                       a1_grid=[1.0], a2_grid=[0.5, 0.75, 1.0],
                       a3_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                       channel_kwargs=channel_kwargs)


def _safe_awss_soft_p1(cfg, res_p1, p_LoS, kappa):
    """Wraps awss_soft(cfg, res_p1) with debug instrumentation. Returns
    (value, was_exception) so the caller can distinguish "legitimately
    infeasible / None" from "the computation blew up silently"."""
    try:
        val = awss_soft(cfg, res_p1)
    except Exception as e:
        if not _AWSS_P1_DEBUG_STATE["printed_exc"]:
            print(f"\n[DEBUG EXCEPTION] awss_soft(cfg, res_p1) raised at "
                  f"p_LoS={p_LoS:.3f}, kappa={kappa:.2f}:")
            import traceback
            traceback.print_exc()
            print("[DEBUG EXCEPTION] (further exceptions in this run will "
                  "be suppressed to avoid log spam)\n")
            _AWSS_P1_DEBUG_STATE["printed_exc"] = True
        return np.nan, True

    # First successful, finite value: confirm the happy path once.
    if not _AWSS_P1_DEBUG_STATE["printed_ok"] and val is not None and np.isfinite(val):
        print(f"[DEBUG OK] awss_soft(P1) succeeded: type={type(val)}, "
              f"value={val} at p_LoS={p_LoS:.3f}, kappa={kappa:.2f}")
        _AWSS_P1_DEBUG_STATE["printed_ok"] = True

    if not _AWSS_P1_DEBUG_STATE["printed_bad"]:
        if val is None or not np.isfinite(val):
            print(f"[DEBUG BAD VALUE] awss_soft(P1) returned non-finite/None "
                  f"without raising: type={type(val)}, value={repr(val)} "
                  f"at p_LoS={p_LoS:.3f}, kappa={kappa:.2f}. "
                  f"res_p1 keys={list(res_p1.keys()) if isinstance(res_p1, dict) else 'N/A'}")
            _AWSS_P1_DEBUG_STATE["printed_bad"] = True

    return val, False


def _exp6_row(p_LoS, kappa, res_p1, res_p3, cfg):
    """Builds one CSV row combining the P1 baseline and the P3 orchestrated
    outcome at a given (p_LoS, kappa) point, including the AWSS_soft gain
    of P3 over P1 and the stage utilizations rho_U/rho_H of the P3
    candidate (used to explain non-monotone AWSS_soft(kappa) trends)."""
    if res_p1 is None:
        awss_p1 = np.nan
        p1_feasible = False
    else:
        awss_p1, _ = _safe_awss_soft_p1(cfg, res_p1, p_LoS, kappa)
        p1_feasible = True

    if res_p3 is None:
        return dict(
            p_LoS=p_LoS, kappa=kappa,
            p1_feasible=p1_feasible, p3_feasible=False,
            alpha=np.nan, B_UH_MHz=np.nan,
            a1=np.nan, a2=np.nan, a3=np.nan,
            rho_U=np.nan, rho_H=np.nan,
            AWSS_soft_P1=awss_p1, AWSS_soft_P3=np.nan,
            AWSS_gain_abs=np.nan, AWSS_gain_pct=np.nan,
            self_throttling_active=np.nan,
        )

    awss_p3 = awss_soft(cfg, res_p3)
    gain_abs = (awss_p3 - awss_p1) if np.isfinite(awss_p1) else np.nan
    gain_pct = (gain_abs / awss_p1 * 100.0) if (np.isfinite(awss_p1) and awss_p1 > 0) else np.nan

    return dict(
        p_LoS=p_LoS, kappa=kappa,
        p1_feasible=p1_feasible, p3_feasible=True,
        alpha=res_p3["alpha"], B_UH_MHz=res_p3["B_UH"] / 1e6,
        a1=res_p3["a"][1], a2=res_p3["a"][2], a3=res_p3["a"][3],
        rho_U=res_p3["rho_U_tot"], rho_H=res_p3["rho_H_tot"],
        AWSS_soft_P1=float(awss_p1) if np.isfinite(awss_p1) else np.nan,
        AWSS_soft_P3=float(awss_p3) if awss_p3 is not None and np.isfinite(awss_p3) else np.nan,
        AWSS_gain_abs=gain_abs, AWSS_gain_pct=gain_pct,
        self_throttling_active=bool(res_p3["a"][3] < 1.0 - 1e-9),
    )


def experiment6_backhaul_degradation():
    print("\n" + "=" * 78)
    print("EXPERIMENT 6: Backhaul degradation (p_LoS sweep) and AWSS survivability")
    print("=" * 78)

    # --- Re-parameterized sweeps (see module docstring) -------------------
    p_LoS_list = np.linspace(0.1, 0.95, 12)
    print(f"[DEBUG] p_LoS_list = {list(p_LoS_list)}")

    # Định nghĩa kappa_list_s5 trước khi chạy debug scan
    kappa_list_s5 = np.concatenate([
        np.linspace(1.0, 3.0, 5),
        np.linspace(3.0, 5.0, 9),
        np.linspace(5.0, 9.0, 5),
    ])
    kappa_list_s5 = np.unique(np.round(kappa_list_s5, 4))  # dedupe boundary points

    # Chạy debug scan kiểm tra tính khả thi
    print("\n[DEBUG] Full feasibility scan across all (p_LoS, kappa):")
    infeasible_count = {p_LoS: 0 for p_LoS in p_LoS_list}
    total_count = {p_LoS: 0 for p_LoS in p_LoS_list}
    for kappa_test in kappa_list_s5:
        for p_LoS in p_LoS_list:
            res_test = _p3_orchestrated_exp6(CFG, kappa_test, dict(p_LoS=p_LoS))
            total_count[p_LoS] += 1
            if res_test is None:
                infeasible_count[p_LoS] += 1
    print("[DEBUG] Infeasible count per p_LoS (out of", len(kappa_list_s5), "kappa values):")
    for p_LoS in p_LoS_list:
        print(f"  p_LoS={p_LoS:.6f} -> {infeasible_count[p_LoS]}/{total_count[p_LoS]} infeasible")

    # ---------------------------------------------------------------- S3 --
    rows_s3 = []
    kappa_s3 = 2.0
    for p_LoS in p_LoS_list:
        ck = dict(p_LoS=p_LoS)
        res_p1 = _p1_baseline_exp6(CFG, kappa_s3, ck)
        res_p3 = _p3_orchestrated_exp6(CFG, kappa_s3, ck)
        row = _exp6_row(p_LoS, kappa_s3, res_p1, res_p3, CFG)
        rows_s3.append(row)
        if row["p3_feasible"]:
            print(f"[S3] p_LoS={p_LoS:.2f}: alpha={row['alpha']:.2f} "
                  f"B_UH={row['B_UH_MHz']:.2f}MHz a={(row['a1'],row['a2'],row['a3'])} "
                  f"rho_U={row['rho_U']:.3f} rho_H={row['rho_H']:.3f} "
                  f"AWSS[P1]={row['AWSS_soft_P1']} AWSS[P3]={row['AWSS_soft_P3']:.3f} "
                  f"gain={row['AWSS_gain_abs']} ({row['AWSS_gain_pct']}%)")
        else:
            print(f"[S3] p_LoS={p_LoS:.2f}: P3 INFEASIBLE "
                  f"(P1 feasible={row['p1_feasible']}, AWSS[P1]={row['AWSS_soft_P1']})")

    df_s3 = pd.DataFrame(rows_s3)

    n_p1_finite_s3 = df_s3["AWSS_soft_P1"].notna().sum()
    print(f"\n[SANITY CHECK] S3: AWSS_soft_P1 is finite in {n_p1_finite_s3}/{len(df_s3)} rows "
          f"({df_s3['p1_feasible'].sum()} rows have p1_feasible=True).")
    if n_p1_finite_s3 == 0 and df_s3["p1_feasible"].sum() > 0:
        print("[SANITY CHECK] *** AWSS_soft_P1 is ALWAYS NaN despite P1 being feasible "
              "in some rows -- this is the bug. Check the [DEBUG ...] lines above for "
              "the root cause (exception vs. silent bad value). ***")

    alpha_vals_s3 = df_s3.loc[df_s3["p3_feasible"], "alpha"]
    alpha_is_pinned_s3 = bool(np.allclose(alpha_vals_s3, ALPHA_MIN_ORCHESTRATION)) if len(alpha_vals_s3) else None
    print(f"[CHECK] alpha pinned at alpha_min={ALPHA_MIN_ORCHESTRATION} across all feasible "
          f"S3 points? {alpha_is_pinned_s3}  (values observed: {sorted(alpha_vals_s3.unique())})")

    _write_csv(df_s3, "exp6a_backhaul_degradation_s3_survivability.csv")

    # ---------------------------------------------------------------- S5 --
    rows_s5 = []
    for kappa in kappa_list_s5:
        for p_LoS in p_LoS_list:
            ck = dict(p_LoS=p_LoS)
            res_p1 = _p1_baseline_exp6(CFG, kappa, ck)
            res_p3 = _p3_orchestrated_exp6(CFG, kappa, ck)
            row = _exp6_row(p_LoS, kappa, res_p1, res_p3, CFG)
            rows_s5.append(row)
            if row["p3_feasible"]:
                print(f"[S5] p_LoS={p_LoS:.2f} kappa={kappa:.2f}: "
                      f"alpha={row['alpha']:.2f} B_UH={row['B_UH_MHz']:.2f}MHz "
                      f"a={(row['a1'],row['a2'],row['a3'])} "
                      f"rho_U={row['rho_U']:.3f} rho_H={row['rho_H']:.3f} "
                      f"throttling={row['self_throttling_active']} "
                      f"AWSS[P1]={row['AWSS_soft_P1']} AWSS[P3]={row['AWSS_soft_P3']}")
            else:
                print(f"[S5] p_LoS={p_LoS:.2f} kappa={kappa:.2f}: P3 INFEASIBLE")

    df_s5 = pd.DataFrame(rows_s5)

    n_p1_finite_s5 = df_s5["AWSS_soft_P1"].notna().sum()
    print(f"\n[SANITY CHECK] S5: AWSS_soft_P1 is finite in {n_p1_finite_s5}/{len(df_s5)} rows "
          f"({df_s5['p1_feasible'].sum()} rows have p1_feasible=True).")

    alpha_vals_s5 = df_s5.loc[df_s5["p3_feasible"], "alpha"]
    alpha_is_pinned_s5 = bool(np.allclose(alpha_vals_s5, ALPHA_MIN_ORCHESTRATION)) if len(alpha_vals_s5) else None
    print(f"[CHECK] alpha pinned at alpha_min={ALPHA_MIN_ORCHESTRATION} across all feasible "
          f"S5 points? {alpha_is_pinned_s5}  (values observed: {sorted(alpha_vals_s5.unique())})")

    p1_infeasible_onset = []
    for p_LoS in p_LoS_list:
        sub = df_s5[(df_s5["p_LoS"] == p_LoS)].sort_values("kappa")
        infeasible_pts = sub[~sub["p1_feasible"]]
        kappa_p1_fail = float(infeasible_pts["kappa"].iloc[0]) if len(infeasible_pts) else np.nan
        p1_infeasible_onset.append(kappa_p1_fail)
    print(f"\n[CHECK] P1 stability boundary (first kappa with rho_U>=1 or rho_H>=1) "
          f"per p_LoS: {p1_infeasible_onset}")

    _write_csv(df_s5, "exp6b_combined_stress_s5_survivability.csv")

    # ------------------------------------------------- self-throttling onset --
    onset_rows = []
    for p_LoS in p_LoS_list:
        sub = df_s5[(df_s5["p_LoS"] == p_LoS) & (df_s5["p3_feasible"])].sort_values("kappa")
        throttled = sub[sub["self_throttling_active"] == True]
        if len(throttled) > 0:
            kappa_onset = float(throttled["kappa"].iloc[0])
            onset_row = throttled.iloc[0]
            onset_rows.append(dict(
                p_LoS=p_LoS,
                kappa_onset=kappa_onset,
                a3_at_onset=onset_row["a3"],
                AWSS_soft_P1_at_onset=onset_row["AWSS_soft_P1"],
                AWSS_soft_P3_at_onset=onset_row["AWSS_soft_P3"],
                AWSS_gain_pct_at_onset=onset_row["AWSS_gain_pct"],
            ))
        else:
            onset_rows.append(dict(
                p_LoS=p_LoS, kappa_onset=np.nan, a3_at_onset=np.nan,
                AWSS_soft_P1_at_onset=np.nan, AWSS_soft_P3_at_onset=np.nan,
                AWSS_gain_pct_at_onset=np.nan,
            ))

    df_onset = pd.DataFrame(onset_rows)
    print("\n-- Self-throttling onset vs. backhaul quality (S5) --")
    print(df_onset.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    valid_onset = df_onset.dropna(subset=["kappa_onset"]).sort_values("p_LoS")
    if len(valid_onset) >= 2:
        onset_vs_pLoS = valid_onset[["p_LoS", "kappa_onset"]].values
        is_monotone_increasing = all(onset_vs_pLoS[i, 1] <= onset_vs_pLoS[i + 1, 1]
                                     for i in range(len(onset_vs_pLoS) - 1))
        print(f"\n[FINDING] Self-throttling onset kappa is "
              f"{'monotonically nondecreasing' if is_monotone_increasing else 'NOT monotone'} "
              f"in p_LoS (worse backhaul -> {'earlier' if is_monotone_increasing else 'unclear'} "
              f"onset of admission throttling).")
    else:
        print("\n[FINDING] Not enough feasible onset points within the tested "
              "kappa range to assess monotonicity.")

    _write_csv(df_onset, "exp6c_self_throttling_onset_vs_pLoS.csv")

    return df_s3, df_s5, df_onset

# ==============================================================================
# EXPERIMENT 7 -- Age-aware admission and the freshness-load tradeoff
# ==============================================================================
def experiment7_freshness_tradeoff(df_exp5: pd.DataFrame):
    """
    Reorganizes the P1-vs-P3 rows ALREADY computed by
    experiment5_policy_comparison() into the freshness-focused table
    described in Sec. XI ("Experiment 7"). No new simulation/analytical
    evaluation is performed here: this guarantees the Exp.7 numbers are
    exactly consistent with Exp.5.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 7: Age-aware admission and the freshness-load tradeoff (P1 vs P3)")
    print("=" * 78)

    sub = df_exp5[df_exp5["policy"].isin(["P1", "P3"])].copy()
    pivot_cols = ["kappa", "policy", "feasible", "PAoI1_s", "AoI_avg1_sim_s",
                  "AGDI", "CSPR", "a3"]
    sub = sub[pivot_cols].copy()
    sub["surrogate_conservative"] = sub["AoI_avg1_sim_s"] <= sub["PAoI1_s"]

    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    n_checked = sub["feasible"].sum()
    n_conservative = sub.loc[sub["feasible"], "surrogate_conservative"].sum()
    print(f"\nPAoI surrogate is conservative (simAoI <= PAoI) in "
          f"{n_conservative}/{n_checked} feasible (kappa,policy) points.")

    _write_csv(sub, "exp7_freshness_tradeoff_p1_vs_p3.csv")
    return sub


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"[INFO] Using N_WORKERS={N_WORKERS} process(es) for seed-level "
          f"parallelism (override with the N_WORKERS environment variable).")

    print("\nOutage probability check (Sec. IX-D):")
    print(f"  baseline (bar_gamma_UH=10, m=2):  P(outage) = "
          f"{outage_probability(10.0, 2.0, 0.1)*100:.4f}%")
    print(f"  pessimistic (bar_gamma_UH=2, m=1): P(outage) = "
          f"{outage_probability(2.0, 1.0, 0.1)*100:.4f}%")

    df1_detail, df1_summary = experiment1_validation()   # bật lại — cần cho Table validation
    df5 = experiment5_policy_comparison()
    df7 = experiment7_freshness_tradeoff(df5)             # bật lại — ăn theo df5
    df2 = experiment2_snr_sweep()
    df3 = experiment3_alpha_sweep()
    df4 = experiment4_bandwidth_split()
    # _plot_bandwidth_split_objective(CFG, {1: 1.0, 2: 1.0, 3: 1.0},
    #                                  CFG.w_delay,
    #                                  os.path.join(FIG_DIR, "exp4_bandwidth_split_objective.png"))
    # import pandas as pd
    # df = pd.read_csv("csv_tables/exp5_policy_comparison_vs_kappa.csv").sort_values(["kappa", "policy"])
    # p3 = df[(df.policy == "P3") & df.feasible]
    # p3 = p3.assign(sampling_term=p3.PAoI1_s - p3.T1_E2E_s,
    #             gap_sim_paoi=p3.PAoI1_s - p3.AoI_avg1_sim_s)
    # print(p3[["kappa", "T1_E2E_s", "PAoI1_s", "sampling_term", "AoI_avg1_sim_s", "gap_sim_paoi"]].to_string(index=False))
    df6a, df6b, df6c = experiment6_backhaul_degradation()
    df7 = experiment7_freshness_tradeoff(df5)

    print("\nAll CSV tables written to:", CSV_DIR)