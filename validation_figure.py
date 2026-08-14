# -*- coding: utf-8 -*-
"""
make_validation_figures.py
================================================================================
Standalone figure generator for Experiment 1 (Sec. IX / IX-D of the paper):
"Validation of the tandem M/G/1 priority approximation".

This script is intentionally SEPARATE from run_experiments.py (which only
computes data and writes CSVs) and from plotting_utils.py (which only sets
the shared Matplotlib style). It READS the CSV already produced by
`experiment1_validation()` and renders figures from it, so:

  - the figures are always in sync with whatever CSV was last generated
    (no risk of a stale, hand-made plot drifting away from the real numbers,
    which is what happened with the previous validation_delay_analysis_
    vs_simulation.pdf that only showed S1 and S4);
  - re-running run_experiments.py + this script reproduces the figures
    from scratch, end to end.

INPUT (required):
    csv_tables/exp1_validation_delay_paoi_detail.csv
    Columns (as written by experiment1_validation() in run_experiments.py):
        scenario, kappa, cls, T_ana_s, T_sim_s, RE_delay_pct,
        PAoI_ana_s, PAoI_sim_s, RE_paoi_pct, rho_U, rho_H, n_seeds_valid

OUTPUT:
    figures/validation_delay_analysis_vs_simulation.pdf   (2x2 grid, S1-S4)
    figures/validation_paoi_analysis_vs_simulation.pdf    (2x2 grid, S1-S4)
    figures/validation_re_delay_paoi_summary.pdf          (bar summary, all 4)

Run:
    python3 make_validation_figures.py
================================================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting_utils import set_academic_style, CLASS_COLORS, CLASS_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "csv_tables", "exp1_validation_delay_paoi_detail.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["scenario", "kappa", "cls", "T_ana_s", "T_sim_s",
                    "RE_delay_pct", "PAoI_ana_s", "PAoI_sim_s", "RE_paoi_pct",
                    "rho_U", "rho_H", "n_seeds_valid"]

# Canonical scenario order (short labels for subplot titles / x-axis).
SCENARIO_ORDER = [
    "S1 (nominal, kappa=1.0)",
    "S2 (weak UAV-HAP link, kappa=1.0)",
    "S3 (degraded backhaul, kappa=1.0)",
    "S4 (disaster surge, kappa=1.6)",
]
SCENARIO_SHORT = {
    "S1 (nominal, kappa=1.0)": "S1 (nominal)",
    "S2 (weak UAV-HAP link, kappa=1.0)": "S2 (weak UH link)",
    "S3 (degraded backhaul, kappa=1.0)": "S3 (degraded backhaul)",
    "S4 (disaster surge, kappa=1.6)": "S4 (surge, \u03ba=1.6)",
}
CLASSES = [1, 2, 3]


def _load_and_validate_csv():
    if not os.path.isfile(CSV_PATH):
        sys.exit(
            f"[error] Could not find:\n  {CSV_PATH}\n"
            f"Run `python3 run_experiments.py` first to generate it "
            f"(experiment1_validation() writes this file)."
        )
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(
            f"[error] {CSV_PATH} is missing expected column(s): {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"This script expects the exact schema written by "
            f"experiment1_validation() in run_experiments.py. If that "
            f"function's output columns changed, update REQUIRED_COLUMNS "
            f"and the plotting code below to match."
        )
    present_scenarios = df["scenario"].unique().tolist()
    missing_scenarios = [s for s in SCENARIO_ORDER if s not in present_scenarios]
    if missing_scenarios:
        print(f"[warn] CSV does not contain scenario(s): {missing_scenarios}. "
              f"Figures will only show the scenarios actually present.")
    return df


def _scenario_list(df):
    """Preserve canonical order, but only for scenarios actually in the CSV."""
    present = df["scenario"].unique().tolist()
    ordered = [s for s in SCENARIO_ORDER if s in present]
    extra = [s for s in present if s not in SCENARIO_ORDER]
    return ordered + extra


# ------------------------------------------------------------------------
# Figure 1: 2x2 grid, class-wise E2E delay, analytical vs simulated
# ------------------------------------------------------------------------
def plot_validation_delay_grid(df):
    scenarios = _scenario_list(df)
    n = len(scenarios)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.6 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    bar_width = 0.35
    x = np.arange(len(CLASSES))

    for i, sname in enumerate(scenarios):
        ax = axes[i]
        sub = df[df["scenario"] == sname].set_index("cls").reindex(CLASSES)

        ana_vals = sub["T_ana_s"].values
        sim_vals = sub["T_sim_s"].values

        for j, k in enumerate(CLASSES):
            ax.bar(x[j] - bar_width / 2, ana_vals[j], width=bar_width,
                   color=CLASS_COLORS[k], alpha=0.55,
                   edgecolor="black", linewidth=0.8,
                   label=f"{CLASS_LABELS[k]} (analytical)" if i == 0 else None)
            ax.bar(x[j] + bar_width / 2, sim_vals[j], width=bar_width,
                   color=CLASS_COLORS[k], alpha=0.95, hatch="//",
                   edgecolor="black", linewidth=0.8,
                   label=f"{CLASS_LABELS[k]} (simulated)" if i == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {k}" for k in CLASSES])
        ax.set_title(SCENARIO_SHORT.get(sname, sname), fontsize=10)
        ax.set_ylabel("Mean E2E delay [s]" if i % ncols == 0 else "")

        re_mean = sub["RE_delay_pct"].mean()
        ax.text(0.98, 0.95, f"mean RE = {re_mean:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.06), frameon=True)
    fig.suptitle("Analytical vs. simulated class-wise E2E delay (S1\u2013S4)",
                 fontsize=12, y=1.12)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "validation_delay_analysis_vs_simulation.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")


# ------------------------------------------------------------------------
# Figure 2: 2x2 grid, class-wise PAoI, analytical vs simulated
# ------------------------------------------------------------------------
def plot_validation_paoi_grid(df):
    scenarios = _scenario_list(df)
    n = len(scenarios)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.6 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    bar_width = 0.35
    x = np.arange(len(CLASSES))

    for i, sname in enumerate(scenarios):
        ax = axes[i]
        sub = df[df["scenario"] == sname].set_index("cls").reindex(CLASSES)

        ana_vals = sub["PAoI_ana_s"].values
        sim_vals = sub["PAoI_sim_s"].values

        for j, k in enumerate(CLASSES):
            ax.bar(x[j] - bar_width / 2, ana_vals[j], width=bar_width,
                   color=CLASS_COLORS[k], alpha=0.55,
                   edgecolor="black", linewidth=0.8,
                   label=f"{CLASS_LABELS[k]} (analytical)" if i == 0 else None)
            ax.bar(x[j] + bar_width / 2, sim_vals[j], width=bar_width,
                   color=CLASS_COLORS[k], alpha=0.95, hatch="//",
                   edgecolor="black", linewidth=0.8,
                   label=f"{CLASS_LABELS[k]} (simulated)" if i == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {k}" for k in CLASSES])
        ax.set_title(SCENARIO_SHORT.get(sname, sname), fontsize=10)
        ax.set_ylabel("Mean peak AoI [s]" if i % ncols == 0 else "")

        re_mean = sub["RE_paoi_pct"].mean()
        ax.text(0.98, 0.95, f"mean RE = {re_mean:.2f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.06), frameon=True)
    fig.suptitle("Analytical vs. simulated class-wise peak AoI (S1\u2013S4)",
                 fontsize=12, y=1.12)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "validation_paoi_analysis_vs_simulation.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")


# ------------------------------------------------------------------------
# Figure 3: single summary bar chart, RE_delay & RE_paoi per scenario
# (mean across the 3 classes), all 4 scenarios side by side.
# ------------------------------------------------------------------------
def plot_re_summary(df):
    scenarios = _scenario_list(df)
    re_delay_mean = [df[df["scenario"] == s]["RE_delay_pct"].mean() for s in scenarios]
    re_delay_max = [df[df["scenario"] == s]["RE_delay_pct"].max() for s in scenarios]
    re_paoi_mean = [df[df["scenario"] == s]["RE_paoi_pct"].mean() for s in scenarios]

    labels = [SCENARIO_SHORT.get(s, s) for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.28

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, re_delay_mean, width=width, label="RE delay (mean)",
           color="#2980b9", edgecolor="black", linewidth=0.8)
    ax.bar(x, re_delay_max, width=width, label="RE delay (max, per class)",
           color="#a9cce3", edgecolor="black", linewidth=0.8, hatch="//")
    ax.bar(x + width, re_paoi_mean, width=width, label="RE PAoI (mean)",
           color="#27ae60", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Analytical-vs-simulation relative error, all validation scenarios")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "validation_re_delay_paoi_summary.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")

    # Also print the global mean/max, matching experiment1_validation()'s
    # own stdout summary, as a sanity cross-check.
    print(f"[check] GLOBAL RE_delay mean={df['RE_delay_pct'].mean():.4f}%  "
          f"max={df['RE_delay_pct'].max():.4f}%")
    print(f"[check] GLOBAL RE_paoi  mean={df['RE_paoi_pct'].mean():.4f}%  "
          f"max={df['RE_paoi_pct'].max():.4f}%")


if __name__ == "__main__":
    set_academic_style()
    df = _load_and_validate_csv()
    print(f"[loaded] {CSV_PATH}  ({len(df)} rows, "
          f"{df['scenario'].nunique()} scenario(s))")

    plot_validation_delay_grid(df)
    plot_validation_paoi_grid(df)
    plot_re_summary(df)

    print("\nAll figures written to:", FIG_DIR)