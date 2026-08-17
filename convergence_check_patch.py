import numpy as np
import pandas as pd
from uav_hap_core import Config, evaluate_analytical
import run_experiments as R  # reuse _simulate_multiseed, CFG, CLASSES, etc.

CFG = R.CFG
CLASSES = R.CLASSES
SIM_HORIZON = R.SIM_HORIZON
SIM_WARMUP = R.SIM_WARMUP
_finite_or_nan = R._finite_or_nan


def _per_seed_re_delay(cfg, ana, kappa, overrides, seed):
    """Run ONE seed and return the mean-over-classes relative delay error
    for that single seed (a genuine i.i.d.-ish sample, not a cumulative
    average)."""
    sim = R._simulate_multiseed(cfg, cfg.alpha, cfg.B_UH, cfg.B_HG,
                                 {1: 1.0, 2: 1.0, 3: 1.0}, kappa=kappa,
                                 horizon=SIM_HORIZON, warmup=SIM_WARMUP,
                                 seeds=(seed,), channel_kwargs=overrides)
    re_list = []
    for k in CLASSES:
        T_ana = _finite_or_nan(ana["T_E2E"][k])
        T_sim = _finite_or_nan(sim[k]["E2E_mean"])
        if T_sim:
            re_list.append(abs(T_ana - T_sim) / T_sim * 100)
    return float(np.mean(re_list)) if re_list else np.nan


def experiment1_seed_convergence_check_all_scenarios(
        scenarios=None,
        max_seeds=1000,
        seeds_per_block=10,
        ci_tol=0.10,        # absolute tolerance on 95% CI half-width, in pct points
        drift_k=1.0,        # multiplier on combined-CI for the "no drift" check
        persist=3,          # consecutive checkpoints required to declare convergence
        min_blocks_for_pass=15,  # refuse to declare convergence before this many blocks
                                  # (guards against low-power false positives at small n)
        seed_base=202601):
    """
    Returns:
      df_blocks   : one row per (scenario, block) -- the raw per-block means,
                    so nothing is hidden inside a smoothed trajectory.
      df_running  : one row per (scenario, n_seeds) -- running mean, SE,
                    95% CI, and the two boolean convergence sub-checks.
      df_verdict  : one row per scenario -- the first n_seeds that passed
                    `persist` consecutive checks (or NOT CONVERGED).
    """
    if scenarios is None:
        scenarios = {
            "S1 (nominal, kappa=1.0)": dict(kappa=1.0, overrides={}),
            "S2 (weak UAV-HAP link, kappa=1.0)": dict(
                kappa=1.0, overrides=dict(bar_gamma_UH=R.S2_BAR_GAMMA_UH)),
            "S3 (degraded backhaul, kappa=1.0)": dict(
                kappa=1.0, overrides=dict(p_LoS=0.3, bar_gamma_HG_LoS=7.5,
                                           bar_gamma_HG_NLoS=2.5)),
            "S4 (disaster surge, kappa=1.6)": dict(kappa=1.6, overrides={}),
        }

    n_blocks_total = max_seeds // seeds_per_block
    seed_pool = [seed_base + i for i in range(max_seeds)]

    block_rows = []
    verdicts = []

    for sname, sk in scenarios.items():
        kappa, overrides = sk["kappa"], sk["overrides"]
        ana = evaluate_analytical(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG,
                                   {1: 1.0, 2: 1.0, 3: 1.0}, kappa=kappa,
                                   **overrides)

        block_means = []
        print(f"\n[{sname}] running {n_blocks_total} blocks of "
              f"{seeds_per_block} seeds each ({max_seeds} seeds total)...")

        for b in range(n_blocks_total):
            block_seeds = seed_pool[b * seeds_per_block:(b + 1) * seeds_per_block]
            per_seed_vals = [
                _per_seed_re_delay(CFG, ana, kappa, overrides, sd)
                for sd in block_seeds
            ]
            per_seed_vals = [v for v in per_seed_vals if np.isfinite(v)]
            if not per_seed_vals:
                continue
            bmean = float(np.mean(per_seed_vals))
            block_means.append(bmean)
            block_rows.append(dict(
                scenario=sname, block_idx=b,
                n_seeds_in_block=len(per_seed_vals),
                block_mean_RE_delay_pct=bmean,
                block_std_RE_delay_pct=float(np.std(per_seed_vals, ddof=1))
                if len(per_seed_vals) > 1 else np.nan,
            ))
            if (b + 1) % 5 == 0 or b == n_blocks_total - 1:
                print(f"  block {b+1:3d}/{n_blocks_total}  "
                      f"n_seeds={( b+1)*seeds_per_block:4d}  "
                      f"block_mean={bmean:.4f}%")

        verdicts.append(_evaluate_convergence(
            sname, block_means, seeds_per_block, ci_tol, drift_k, persist,
            min_blocks_for_pass=min_blocks_for_pass))

    df_blocks = pd.DataFrame(block_rows)

    df_running = pd.concat(
        [v["running_df"] for v in verdicts], ignore_index=True
    ) if verdicts else pd.DataFrame()

    df_verdict = pd.DataFrame([
        {k: v[k] for k in
         ("scenario", "converged", "n_seeds_converged",
          "mean_at_convergence_pct", "ci95_halfwidth_at_convergence_pct",
          "best_available_n_seeds", "best_available_mean_pct",
          "best_available_ci95_halfwidth_pct")}
        for v in verdicts
    ])

    R._write_csv(df_blocks, "exp1_convergence_block_raw.csv")
    R._write_csv(df_running, "exp1_convergence_running_stats.csv")
    R._write_csv(df_verdict, "exp1_convergence_verdict.csv")

    print("\n" + "=" * 78)
    print("CONVERGENCE VERDICT (evidence-based, not eyeballed)")
    print("=" * 78)
    print(df_verdict.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Diagnostic: for scenarios that did NOT converge, show exactly which
    # sub-check (tol_pass vs drift_pass) was failing at the last checkpoint,
    # so it's clear whether the bottleneck is "CI still too wide" or
    # "still statistically drifting relative to n/2".
    print("\n" + "=" * 78)
    print("DIAGNOSTIC: last-checkpoint sub-check status per scenario")
    print("(tol_pass = CI95 half-width < ci_tol; "
          "drift_pass = mean(n) vs mean(n/2) consistent; "
          "trend_pass = no monotone run in recent block means)")
    print("=" * 78)
    if not df_running.empty:
        last_rows = (df_running.sort_values("n_seeds")
                     .groupby("scenario").tail(1))
        print(last_rows[["scenario", "n_seeds", "running_mean_RE_delay_pct",
                          "ci95_halfwidth_pct", "enough_data", "tol_pass",
                          "drift_pass", "trend_pass", "overall_pass"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return df_blocks, df_running, df_verdict


def _evaluate_convergence(sname, block_means, seeds_per_block, ci_tol,
                           drift_k, persist, min_blocks_for_pass=15):
    """Given the sequence of block means (each block = seeds_per_block
    fresh seeds), compute the running mean/SE/CI at every checkpoint and
    apply the two-part statistical convergence test with a `persist`
    consecutive-pass requirement.

    IMPORTANT FIX: at low block counts the half-point drift test
    (mean(n) vs mean(n/2)) has very low statistical power -- both CIs are
    wide, so "not significantly different" is trivially true even when the
    process is nowhere near stable (this produced a false "converged at
    n=50" result for S1 in an earlier run, where the raw block means were
    still swinging by ~0.7 percentage points at n=600). We therefore:
      (a) refuse to declare a pass before `min_blocks_for_pass` blocks
          have been collected, regardless of what the CI/drift test says,
          and
      (b) additionally require that there is no monotonic trend in the
          last `persist`+2 block means (a simple sign-based trend check),
          so a CI-based "no significant difference" verdict cannot hide a
          slow, still-directional drift.
    """
    running_rows = []
    n_blocks = len(block_means)
    passes = []  # boolean pass/fail history, for the persistence check

    for i in range(1, n_blocks + 1):
        used = block_means[:i]
        n_seeds = i * seeds_per_block
        mean_i = float(np.mean(used))
        se_i = float(np.std(used, ddof=1) / np.sqrt(i)) if i > 1 else np.nan
        ci_i = 1.96 * se_i if np.isfinite(se_i) else np.nan

        tol_pass = np.isfinite(ci_i) and (ci_i < ci_tol)

        # Half-point drift check: compare running mean at i blocks against
        # the running mean at i//2 blocks (an earlier, independent-ish
        # estimate). If they don't overlap within the combined CI, we are
        # still drifting. Only meaningful once i is large enough that both
        # halves have decent power -- gated below by min_blocks_for_pass.
        half = max(1, i // 2)
        if half < i:
            used_half = block_means[:half]
            mean_half = float(np.mean(used_half))
            se_half = (float(np.std(used_half, ddof=1) / np.sqrt(half))
                       if half > 1 else np.nan)
            ci_half = 1.96 * se_half if np.isfinite(se_half) else np.nan
            if np.isfinite(ci_i) and np.isfinite(ci_half):
                combined = drift_k * (ci_i + ci_half)
                drift_pass = abs(mean_i - mean_half) < combined
            else:
                drift_pass = False
        else:
            drift_pass = False  # not enough history yet

        # Trend check: over the last `persist`+2 raw block means (not the
        # running mean), require that they are NOT monotonically
        # increasing or decreasing. A monotone run is weak evidence of
        # residual drift that a wide-CI pairwise test can miss entirely.
        window = block_means[max(0, i - (persist + 2)):i]
        if len(window) >= 4:
            diffs = np.diff(window)
            trend_pass = not (np.all(diffs > 0) or np.all(diffs < 0))
        else:
            trend_pass = True  # not enough points yet to judge a trend

        enough_data = i >= min_blocks_for_pass
        overall_pass = bool(enough_data and tol_pass and drift_pass and trend_pass)
        passes.append(overall_pass)

        running_rows.append(dict(
            scenario=sname, n_seeds=n_seeds, n_blocks_used=i,
            running_mean_RE_delay_pct=mean_i,
            SE_pct=se_i, ci95_halfwidth_pct=ci_i,
            enough_data=enough_data,
            tol_pass=tol_pass, drift_pass=drift_pass, trend_pass=trend_pass,
            overall_pass=overall_pass,
        ))

    running_df = pd.DataFrame(running_rows)

    # Find first index where `persist` consecutive passes occur.
    converged_n = None
    converged_mean = None
    converged_ci = None
    for i in range(len(passes) - persist + 1):
        if all(passes[i:i + persist]):
            converged_n = running_rows[i]["n_seeds"]
            converged_mean = running_rows[i]["running_mean_RE_delay_pct"]
            converged_ci = running_rows[i]["ci95_halfwidth_pct"]
            break

    best_row = running_rows[-1] if running_rows else dict(
        n_seeds=np.nan, running_mean_RE_delay_pct=np.nan,
        ci95_halfwidth_pct=np.nan)

    return dict(
        scenario=sname,
        converged=converged_n is not None,
        n_seeds_converged=converged_n if converged_n is not None else np.nan,
        mean_at_convergence_pct=converged_mean if converged_mean is not None else np.nan,
        ci95_halfwidth_at_convergence_pct=converged_ci if converged_ci is not None else np.nan,
        best_available_n_seeds=best_row["n_seeds"],
        best_available_mean_pct=best_row["running_mean_RE_delay_pct"],
        best_available_ci95_halfwidth_pct=best_row["ci95_halfwidth_pct"],
        running_df=running_df,
    )


def physical_diagnostics_per_scenario(df_blocks: pd.DataFrame, scenarios=None):
    """
    Cross-checks the two candidate physical explanations for why some
    scenarios show larger block-to-block variance in RE_delay than others:

      Hypothesis A (S2, weak link): a low mean SNR pushes probability mass
      of gamma close to gamma_min, where 1/ln(1+gamma) is steep and convex.
      This inflates E[S^2] disproportionately relative to E[S] -- i.e. a
      HEAVY-TAILED service-time distribution -- which should show up as a
      large coefficient of variation CV(S) = sqrt(Var(S))/E(S) for the
      UAV-HAP hop, even though rho_U itself may not be close to 1.

      Hypothesis B (S4, surge): E[W_i^X] = W0^X / [(1-rho_{1:i})(1-rho_{1:i-1})].
      As kappa pushes rho^U toward 1, the denominator shrinks, so any given
      amount of noise in W0 or rho gets AMPLIFIED into much larger noise in
      W_i. This is a near-instability amplification effect, distinct from
      the heavy-tail mechanism of Hypothesis A, and should show up as
      rho_U (or rho_H) being close to 1, even if CV(S) itself is modest.

    This function does NOT assume either hypothesis is correct: it just
    tabulates the physical quantities (rho_U, rho_H, CV of service time
    per class, W0) alongside the EMPIRICAL block-to-block std of RE_delay
    already collected in df_blocks, so the two can be inspected side by
    side and the correlation (if any) checked directly rather than argued
    from formulas alone.
    """
    if scenarios is None:
        scenarios = {
            "S1 (nominal, kappa=1.0)": dict(kappa=1.0, overrides={}),
            "S2 (weak UAV-HAP link, kappa=1.0)": dict(
                kappa=1.0, overrides=dict(bar_gamma_UH=R.S2_BAR_GAMMA_UH)),
            "S3 (degraded backhaul, kappa=1.0)": dict(
                kappa=1.0, overrides=dict(p_LoS=0.3, bar_gamma_HG_LoS=7.5,
                                           bar_gamma_HG_NLoS=2.5)),
            "S4 (disaster surge, kappa=1.6)": dict(kappa=1.6, overrides={}),
        }

    a = {1: 1.0, 2: 1.0, 3: 1.0}
    rows = []

    for sname, sk in scenarios.items():
        kappa, overrides = sk["kappa"], sk["overrides"]
        ana = evaluate_analytical(CFG, CFG.alpha, CFG.B_UH, CFG.B_HG, a,
                                   kappa=kappa, **overrides)

        rho_U = _finite_or_nan(ana["rho_U_tot"])
        rho_H = _finite_or_nan(ana["rho_H_tot"])

        # Per-class, per-hop service-time moments -> coefficient of
        # variation. `ana["mom_U"][k]` / `ana["mom_H"][k]` are expected to
        # be (E[S], E[S^2]) tuples per the existing usage in
        # run_experiments.experiment3_alpha_sweep (W_U = T_U - mom_U[k][0]).
        cv_rows = {}
        for hop_name, mom_key in (("UH", "mom_U"), ("HG", "mom_H")):
            for k in CLASSES:
                try:
                    E_S, E_S2 = ana[mom_key][k]
                    E_S = _finite_or_nan(E_S)
                    E_S2 = _finite_or_nan(E_S2)
                    var_S = E_S2 - E_S ** 2 if np.isfinite(E_S2) and np.isfinite(E_S) else np.nan
                    cv = (np.sqrt(var_S) / E_S
                          if np.isfinite(var_S) and var_S >= 0 and E_S else np.nan)
                except (KeyError, TypeError, ValueError):
                    cv = np.nan
                cv_rows[f"CV_S_{hop_name}_class{k}"] = cv

        # W0 residual-service term per hop, if exposed by evaluate_analytical.
        W0_U = _finite_or_nan(ana.get("W0_U", np.nan))
        W0_H = _finite_or_nan(ana.get("W0_H", np.nan))

        # Empirical block-to-block variance of RE_delay for this scenario,
        # taken from the SAME blocks already computed by
        # experiment1_seed_convergence_check_all_scenarios (no recomputation).
        sub = df_blocks[df_blocks["scenario"] == sname]
        block_means = sub["block_mean_RE_delay_pct"].dropna().values
        empirical_block_std = (float(np.std(block_means, ddof=1))
                                if len(block_means) > 1 else np.nan)
        empirical_block_cv = (empirical_block_std / float(np.mean(block_means))
                               if len(block_means) > 1 and np.mean(block_means)
                               else np.nan)

        row = dict(
            scenario=sname, kappa=kappa,
            rho_U=rho_U, rho_H=rho_H,
            one_minus_rho_U=(1 - rho_U) if np.isfinite(rho_U) else np.nan,
            one_minus_rho_H=(1 - rho_H) if np.isfinite(rho_H) else np.nan,
            W0_U=W0_U, W0_H=W0_H,
            empirical_block_std_RE_delay_pct=empirical_block_std,
            empirical_block_CV_RE_delay=empirical_block_cv,
        )
        row.update(cv_rows)
        rows.append(row)

    df_phys = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("PHYSICAL-MECHANISM DIAGNOSTICS")
    print("(rho close to 1 -> near-instability amplification hypothesis;")
    print(" large CV_S -> heavy-tailed service-time hypothesis;")
    print(" compare against empirical_block_std_RE_delay_pct, the actual")
    print(" observed block-to-block noise in the convergence-check data)")
    print("=" * 78)
    display_cols = (["scenario", "rho_U", "rho_H"]
                     + [c for c in df_phys.columns if c.startswith("CV_S_UH")]
                     + ["empirical_block_std_RE_delay_pct"])
    display_cols = [c for c in display_cols if c in df_phys.columns]
    print(df_phys[display_cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # Rank scenarios by empirical noise vs. by rho_U vs. by CV, so the
    # two hypotheses can be checked against the actual ranking instead of
    # just eyeballing the table.
    if df_phys["empirical_block_std_RE_delay_pct"].notna().any():
        rank_noise = df_phys.sort_values(
            "empirical_block_std_RE_delay_pct", ascending=False
        )["scenario"].tolist()
        rank_rho = df_phys.sort_values(
            "rho_U", ascending=False)["scenario"].tolist()
        cv_uh_cols = [c for c in df_phys.columns if c.startswith("CV_S_UH")]
        if cv_uh_cols:
            df_phys["CV_S_UH_max_over_classes"] = df_phys[cv_uh_cols].max(axis=1)
            rank_cv = df_phys.sort_values(
                "CV_S_UH_max_over_classes", ascending=False)["scenario"].tolist()
        else:
            rank_cv = []
        print("\nRanking by EMPIRICAL block-to-block noise (most to least):")
        print("  " + " > ".join(rank_noise))
        print("Ranking by rho_U (most to least saturated):")
        print("  " + " > ".join(rank_rho))
        if rank_cv:
            print("Ranking by max CV(S) on UAV-HAP hop (most to least heavy-tailed):")
            print("  " + " > ".join(rank_cv))
        print("\nIf the empirical-noise ranking matches the rho_U ranking, "
              "that supports the near-instability-amplification hypothesis.")
        print("If it matches the CV(S) ranking instead, that supports the "
              "heavy-tailed-service-time hypothesis. If it matches neither, "
              "neither single-mechanism story is sufficient and both "
              "effects (or a different one) are likely mixing.")

    R._write_csv(df_phys, "exp1_physical_mechanism_diagnostics.csv")
    return df_phys


if __name__ == "__main__":
    _, df_running, _ = experiment1_seed_convergence_check_all_scenarios()
    df_blocks_reload = pd.read_csv(
        __import__("os").path.join(R.CSV_DIR, "exp1_convergence_block_raw.csv"))
    physical_diagnostics_per_scenario(df_blocks_reload)