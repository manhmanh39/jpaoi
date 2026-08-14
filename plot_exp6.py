# -*- coding: utf-8 -*-
"""
plot_exp6.py
================================================================================
Generates the figures for Experiment 6 (backhaul degradation S3 / combined
stress S5 survivability) from the CSV tables written by
run_experiments.py::experiment6_backhaul_degradation().

REVISION NOTES (this version, on top of the previous overlay revision)
------------------------------------------------------------------------
This revision adapts the figures to the ACTUAL shape of the re-run Exp.6
data, which has two properties the previous version did not account for:

  (a) kappa is NOT evenly spaced (0.5-steps up to 2.5, then 0.25-steps up
      to 5.0, then 1.0-steps up to 9.0). This means:
        - Fig.4's heatmap must NOT use imshow() with a linear extent
          (which silently assumes a regular grid and would visually
          distort the 3.0-5.0 region where samples are denser). It now
          uses pcolormesh() with the TRUE kappa edges, so cell heights on
          the plot are proportional to the actual kappa spacing.
        - Fig.2's line plots already handle uneven x-spacing correctly
          (matplotlib just connects the actual data points), so no change
          needed there beyond what's described below.

  (b) At kappa=2.0 P1 is still feasible everywhere, but by kappa=3.0 P1 is
      already infeasible for low p_LoS and only marginally feasible for
      high p_LoS (see exp6b: p1_feasible=False for most p_LoS at
      kappa=3.0-9.0). This is a genuinely new, citable finding (P1's
      collapse threshold in kappa is itself pulled earlier by degraded
      backhaul) that the single-kappa-slice Fig.1 (kappa=2.0 only) could
      not show. Fig.1 is now a TWO-PANEL figure: kappa=2.0 (P1 fully
      feasible, matches the original text) side-by-side with kappa=3.0
      (P1 partially collapsed), so both the "moderate stress: resource
      reallocation suffices" and "higher stress: P1 already fails while
      P3 is still feasible" findings are visible without needing a new
      figure elsewhere.
      Points where P1 is infeasible are now marked explicitly (rather
      than silently producing a broken/absent line) with an "X P1
      infeasible" marker, so the reader isn't left wondering why the red
      line stops.

  (c) Fig.3's onset-vs-p_LoS curve is now visibly a STEP function with the
      coarser 12-point p_LoS grid (several consecutive p_LoS share the
      same kappa_onset). It is now drawn with drawstyle="steps-mid"
      instead of a naive line-through-points, which correctly conveys
      "onset is constant over this p_LoS range, then jumps" instead of
      implying a smooth interpolation between plateaus. Annotations are
      de-duplicated: only the first point of each plateau and the last
      point overall are labelled with a3_at_onset, avoiding label
      clutter from repeated identical values.

Fig.2 keeps the single-panel multi-line overlay from the previous revision
(already appropriate for a ~12-point p_LoS grid). Fig.4 keeps the heatmap
default but switches imshow -> pcolormesh for correct non-uniform-kappa
rendering; the raw line-overlay alternative (plot_fig4_rho_vs_kappa) is
unchanged and still available as a fallback.

Reads:
  csv_tables/exp6a_backhaul_degradation_s3_survivability.csv
  csv_tables/exp6b_combined_stress_s5_survivability.csv
  csv_tables/exp6c_self_throttling_onset_vs_pLoS.csv

Writes (PDF, matching the paper's figure style):
  figures/exp6_fig1_awss_vs_pLoS_s3_twopanel.pdf
  figures/exp6_fig2_awss_vs_kappa_s5_overlay.pdf
  figures/exp6_fig3_throttling_onset_vs_pLoS_steps.pdf
  figures/exp6_fig4_rho_gap_heatmap_s5.pdf

Run:  python3 plot_exp6.py
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_DIR = os.path.join(os.path.dirname(__file__), "csv_tables")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "figure.figsize": (6.0, 4.2),
})

COLORS = {
    "P1": "tab:red",
    "P3": "tab:blue",
    "rho_U": "tab:blue",
    "rho_H": "tab:red",
}


# ------------------------------------------------------------------------
# Figure 1 -- S3: AWSS_soft(P1) vs AWSS_soft(P3) versus p_LoS
#   TWO PANELS: kappa=2.0 (P1 fully feasible) and kappa=3.0 (P1 partially
#   collapsed). Infeasible-P1 points are marked explicitly rather than
#   silently vanishing.
#   X: p_LoS      Y: AWSS_soft_P1, AWSS_soft_P3
# ------------------------------------------------------------------------
def plot_fig1_s3_awss_vs_pLoS(kappa_panels=(2.0, 3.0)):
    df = pd.read_csv(os.path.join(CSV_DIR, "exp6a_backhaul_degradation_s3_survivability.csv"))

    # exp6a as generated only holds kappa=2.0 by construction (kappa_s3 is
    # fixed in run_experiments.py). If a second kappa is requested here,
    # pull it from exp6b (S5), which sweeps kappa x p_LoS and contains
    # kappa=2.0 and kappa=3.0 as a subset.
    df_s5 = pd.read_csv(os.path.join(CSV_DIR, "exp6b_combined_stress_s5_survivability.csv"))

    fig, axes = plt.subplots(1, len(kappa_panels), figsize=(11.5, 4.4), sharey=False)
    if len(kappa_panels) == 1:
        axes = [axes]

    for ax, kappa in zip(axes, kappa_panels):
        if abs(kappa - 2.0) < 1e-9 and not df.empty:
            sub = df.sort_values("p_LoS")
        else:
            sub = df_s5[np.isclose(df_s5["kappa"], kappa)].sort_values("p_LoS")

        feas_p1 = sub[sub["p1_feasible"] == True]
        infeas_p1 = sub[sub["p1_feasible"] == False]

        ax.plot(feas_p1["p_LoS"], feas_p1["AWSS_soft_P1"], "o-",
                color=COLORS["P1"], label="P1 (fixed, no admission)")
        if len(infeas_p1) > 0:
            ax.plot(infeas_p1["p_LoS"], np.zeros(len(infeas_p1)), "x",
                    color=COLORS["P1"], markersize=9, markeredgewidth=2,
                    label="P1 infeasible" if ax is axes[0] else None,
                    clip_on=False, zorder=5)

        ax.plot(sub["p_LoS"], sub["AWSS_soft_P3"], "s-", color=COLORS["P3"],
                label="P3 (proposed orchestration)")

        for _, row in sub.iterrows():
            if np.isfinite(row.get("AWSS_gain_pct", np.nan)) and row["p1_feasible"]:
                ax.annotate(f"+{row['AWSS_gain_pct']:.0f}%",
                            (row["p_LoS"], row["AWSS_soft_P3"]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=COLORS["P3"])

        ax.set_xlabel(r"LoS probability $p_{\mathrm{LoS}}$")
        ax.set_title(rf"$\kappa={kappa:.1f}$")

    axes[0].set_ylabel(r"Soft weighted service survivability, $\mathrm{AWSS_{soft}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle("S3: backhaul degradation -- moderate vs. elevated surge", y=1.14, fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp6_fig1_awss_vs_pLoS_s3_twopanel.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 2 -- S5: AWSS_soft(P1) vs AWSS_soft(P3) versus kappa.
#   Single-panel overlay, one color per p_LoS (~12 points -> tab20 is
#   legible). Kappa spacing is uneven (denser 3.0-5.0) -- matplotlib
#   connects actual data points so this is handled correctly without
#   extra work; only the P1-infeasible endpoint needs marking.
#   X: kappa      Y: AWSS_soft_P1, AWSS_soft_P3     grouped by: p_LoS
# ------------------------------------------------------------------------
def plot_fig2_s5_awss_vs_kappa(subplots=False):
    df = pd.read_csv(os.path.join(CSV_DIR, "exp6b_combined_stress_s5_survivability.csv"))
    p_LoS_list = sorted(df["p_LoS"].unique())

    if not subplots:
        n = len(p_LoS_list)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(n, 2)))

        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        for i, p_LoS in enumerate(p_LoS_list):
            sub = df[df["p_LoS"] == p_LoS].sort_values("kappa")
            c = cmap[i]
            ax.plot(sub["kappa"], sub["AWSS_soft_P3"], "-", color=c,
                    linewidth=1.6,
                    label=rf"$p_{{\mathrm{{LoS}}}}={p_LoS:.2f}$")

            feas_p1 = sub[sub["p1_feasible"] == True]
            ax.plot(feas_p1["kappa"], feas_p1["AWSS_soft_P1"], "--", color=c,
                    linewidth=1.1, alpha=0.55)
            # Mark where P1 becomes infeasible along this p_LoS curve.
            infeas_p1 = sub[sub["p1_feasible"] == False]
            if len(infeas_p1) > 0:
                kappa_fail = infeas_p1["kappa"].min()
                ax.axvline(kappa_fail, color=c, linewidth=0.6, alpha=0.25, zorder=0)

        style_handles = [
            Line2D([0], [0], color="gray", linestyle="-", linewidth=1.6, label="P3 (solid)"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.1, alpha=0.55,
                   label="P1 (dashed, until infeasible)"),
        ]
        color_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                  fontsize=8, title=r"$p_{\mathrm{LoS}}$", ncol=1)
        ax.add_artist(color_legend)
        ax.legend(handles=style_handles, loc="upper left",
                  bbox_to_anchor=(1.02, 0.30), fontsize=9)

        ax.set_xlabel(r"Surge factor $\kappa$")
        ax.set_ylabel(r"$\mathrm{AWSS_{soft}}$")
        ax.set_title("S5: combined stress -- P1 vs P3 across backhaul quality\n"
                      "(thin vertical ticks mark where P1 becomes infeasible)",
                      fontsize=10)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exp6_fig2_awss_vs_kappa_s5_overlay.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")

    else:
        n = len(p_LoS_list)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 2.6 * nrows),
                                  sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for i, p_LoS in enumerate(p_LoS_list):
            ax = axes[i]
            sub = df[df["p_LoS"] == p_LoS].sort_values("kappa")
            ax.plot(sub["kappa"], sub["AWSS_soft_P1"], "-", color=COLORS["P1"],
                    linewidth=1.0, label="P1" if i == 0 else None)
            ax.plot(sub["kappa"], sub["AWSS_soft_P3"], "-", color=COLORS["P3"],
                    linewidth=1.0, label="P3" if i == 0 else None)
            ax.set_title(rf"$p_{{\mathrm{{LoS}}}}={p_LoS:.2f}$", fontsize=9)
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel(r"$\kappa$", fontsize=9)
            if i % ncols == 0:
                ax.set_ylabel(r"$\mathrm{AWSS_{soft}}$", fontsize=9)

        for j in range(len(p_LoS_list), len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exp6_fig2_awss_vs_kappa_s5_grid.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 3 -- Self-throttling onset kappa versus p_LoS
#   Drawn as a step function (drawstyle="steps-mid") to correctly convey
#   the plateau structure now visible with the coarser 12-point p_LoS
#   grid, instead of implying smooth interpolation between samples.
#   Annotations are de-duplicated: only label the first point of each
#   plateau (a3_at_onset change) plus the last point overall.
#   X: p_LoS      Y: kappa_onset
# ------------------------------------------------------------------------
def plot_fig3_throttling_onset():
    df = pd.read_csv(os.path.join(CSV_DIR, "exp6c_self_throttling_onset_vs_pLoS.csv"))
    df = df.sort_values("p_LoS").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(df["p_LoS"], df["kappa_onset"], "o", color="tab:purple",
            markersize=5, zorder=3)
    ax.step(df["p_LoS"], df["kappa_onset"], where="mid", color="tab:purple",
            linewidth=1.6, zorder=2)

    # Label only the first sample of each plateau (kappa_onset value
    # change) and the last row overall, to avoid repeating the same
    # a3_at_onset annotation across several consecutive p_LoS points.
    prev_val = None
    for i, row in df.iterrows():
        is_new_plateau = (prev_val is None) or (not np.isclose(row["kappa_onset"], prev_val))
        is_last = (i == len(df) - 1)
        if np.isfinite(row.get("a3_at_onset", np.nan)) and (is_new_plateau or is_last):
            ax.annotate(rf"$a_3$={row['a3_at_onset']:.2f}",
                        (row["p_LoS"], row["kappa_onset"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)
        prev_val = row["kappa_onset"]

    ax.set_xlabel(r"LoS probability $p_{\mathrm{LoS}}$")
    ax.set_ylabel(r"Self-throttling onset $\kappa_{\mathrm{onset}}$")
    ax.set_title("Onset of admission throttling vs. backhaul quality")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp6_fig3_throttling_onset_vs_pLoS_steps.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 4 -- P3 stage-utilization balance across the (p_LoS, kappa) plane.
#   Heatmap of |rho_U - rho_H| using pcolormesh with the TRUE (non-uniform)
#   kappa edges, so cell sizes on the plot reflect the actual sampling
#   density (denser in the 3.0-5.0 range) instead of being silently
#   stretched to a regular grid as imshow() would do.
#   X: p_LoS   Y: kappa   COLOR: |rho_U - rho_H|
# ------------------------------------------------------------------------
def _edges_from_centers(centers):
    """Builds cell edges for pcolormesh from a sorted array of (possibly
    non-uniform) cell centers, using midpoints and extrapolating the two
    outer edges symmetrically."""
    centers = np.asarray(centers, dtype=float)
    mids = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_fig4_rho_gap_heatmap():
    df = pd.read_csv(os.path.join(CSV_DIR, "exp6b_combined_stress_s5_survivability.csv"))
    df = df[df["p3_feasible"]].copy()
    df["rho_gap_abs"] = (df["rho_U"] - df["rho_H"]).abs()

    pivot = df.pivot_table(index="kappa", columns="p_LoS", values="rho_gap_abs")
    pivot = pivot.sort_index().sort_index(axis=1)

    x_edges = _edges_from_centers(pivot.columns.values)
    y_edges = _edges_from_centers(pivot.index.values)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    mesh = ax.pcolormesh(x_edges, y_edges, pivot.values, cmap="magma",
                          shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$|\rho^U - \rho^H|$ (P3 candidate)")

    ax.set_xlabel(r"LoS probability $p_{\mathrm{LoS}}$")
    ax.set_ylabel(r"Surge factor $\kappa$ (non-uniform grid, true spacing shown)")
    ax.set_title("P3 stage-utilization balance across surge and backhaul quality")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exp6_fig4_rho_gap_heatmap_s5.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_fig4_rho_vs_kappa(subplots=False):
    """Line-overlay alternative to the Fig.4 heatmap, kept for anyone who
    wants the raw rho_U/rho_H curves instead of the balance-gap heatmap."""
    df = pd.read_csv(os.path.join(CSV_DIR, "exp6b_combined_stress_s5_survivability.csv"))
    p_LoS_list = sorted(df["p_LoS"].unique())

    if not subplots:
        n = len(p_LoS_list)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(n, 2)))

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for i, p_LoS in enumerate(p_LoS_list):
            sub = df[(df["p_LoS"] == p_LoS) & df["p3_feasible"]].sort_values("kappa")
            c = cmap[i]
            ax.plot(sub["kappa"], sub["rho_U"], "-", color=c, linewidth=1.4,
                    label=rf"$p_{{\mathrm{{LoS}}}}={p_LoS:.2f}$")
            ax.plot(sub["kappa"], sub["rho_H"], "--", color=c, linewidth=1.1,
                    alpha=0.6)

        ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")

        style_handles = [
            Line2D([0], [0], color="gray", linestyle="-", linewidth=1.4, label=r"$\rho^U$ (solid)"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.1, alpha=0.6, label=r"$\rho^H$ (dashed)"),
        ]
        color_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                  fontsize=8, title=r"$p_{\mathrm{LoS}}$")
        ax.add_artist(color_legend)
        ax.legend(handles=style_handles, loc="upper left",
                  bbox_to_anchor=(1.02, 0.35), fontsize=9)

        ax.set_xlabel(r"Surge factor $\kappa$")
        ax.set_ylabel(r"Stage utilization $\rho$")
        ax.set_title("P3 stage utilization vs. surge and backhaul quality")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exp6_fig4_rho_vs_kappa_s5_overlay.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")

    else:
        n = len(p_LoS_list)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 2.6 * nrows),
                                  sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for i, p_LoS in enumerate(p_LoS_list):
            ax = axes[i]
            sub = df[(df["p_LoS"] == p_LoS) & df["p3_feasible"]].sort_values("kappa")
            ax.plot(sub["kappa"], sub["rho_U"], "-", color=COLORS["rho_U"],
                    linewidth=1.0, label=r"$\rho^U$" if i == 0 else None)
            ax.plot(sub["kappa"], sub["rho_H"], "--", color=COLORS["rho_H"],
                    linewidth=1.0, label=r"$\rho^H$" if i == 0 else None)
            ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
            ax.set_title(rf"$p_{{\mathrm{{LoS}}}}={p_LoS:.2f}$", fontsize=9)
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel(r"$\kappa$", fontsize=9)
            if i % ncols == 0:
                ax.set_ylabel(r"$\rho$", fontsize=9)

        for j in range(len(p_LoS_list), len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exp6_fig4_rho_vs_kappa_s5_grid.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    plot_fig1_s3_awss_vs_pLoS(kappa_panels=(2.0, 3.0))
    plot_fig2_s5_awss_vs_kappa(subplots=False)
    plot_fig3_throttling_onset()
    plot_fig4_rho_gap_heatmap()
    print("\nAll Experiment 6 figures written to:", FIG_DIR)