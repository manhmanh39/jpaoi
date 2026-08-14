# -*- coding: utf-8 -*-
"""
plot_exp5_exp7.py
================================================================================
Vẽ các hình cho Experiment 5 (P1 vs P2 vs P3 policy comparison) và
Experiment 7 (freshness-load tradeoff, P1 vs P3), đọc trực tiếp từ:

    csv_tables/exp5_policy_comparison_vs_kappa.csv

(exp7 được suy ra ngay từ exp5 -- không cần file exp7 riêng, đúng như
logic gốc của experiment7_freshness_tradeoff() chỉ lọc lại các cột của
exp5, không tính toán gì mới).

Xuất ra (PDF, để cùng thư mục figures/):
    figures/fig5a_T1_vs_kappa.pdf         -- Class-1 delay vs kappa (3 policy)
    figures/fig5b_cspr_vs_kappa.pdf       -- CSPR vs kappa (3 policy)
    figures/fig5c_gdi_vs_kappa.pdf        -- GDI vs kappa (3 policy)
    figures/fig5d_a3_vs_kappa.pdf         -- admission ratio a3 vs kappa (P3 only)
    figures/fig7a_paoi_vs_kappa.pdf       -- Class-1 PAoI vs kappa (P1 vs P3)
    figures/fig7b_avg_aoi_sim_vs_kappa.pdf-- simulated avg AoI vs kappa (P1 vs P3)
    figures/fig7c_surrogate_gap_vs_kappa.pdf -- PAoI surrogate - simAoI gap (P3)

Run:  python3 plot_exp5_exp7.py
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_tables")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "figure.figsize": (6.2, 4.2),
})

COLORS = {"P1": "tab:red", "P2": "tab:green", "P3": "tab:blue"}
MARKERS = {"P1": "o", "P2": "^", "P3": "s"}


def _load_exp5():
    path = os.path.join(CSV_DIR, "exp5_policy_comparison_vs_kappa.csv")
    df = pd.read_csv(path)
    # feasible may come in as True/False strings or bools depending on how
    # the CSV was written; normalize to bool.
    if df["feasible"].dtype == object:
        df["feasible"] = df["feasible"].astype(str).str.strip().str.lower() == "true"
    return df


# ------------------------------------------------------------------------
# Figure 5a -- Class-1 end-to-end delay vs kappa, 3 policies
#   X: kappa      Y: T1_E2E_s      lines: policy
# ------------------------------------------------------------------------
def plot_fig5a_delay_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P2", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        ax.plot(sub["kappa"], sub["T1_E2E_s"], marker=MARKERS[p],
                color=COLORS[p], label=p, markersize=5, linewidth=1.4)
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel(r"Class-1 end-to-end delay $T_1^{\mathrm{E2E}}$ [s]")
    ax.set_title("Class-1 delay vs. surge: P1 vs P2 vs P3")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5a_T1_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 5b -- CSPR vs kappa, 3 policies
#   X: kappa      Y: CSPR      lines: policy
# ------------------------------------------------------------------------
def plot_fig5b_cspr_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P2", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        ax.plot(sub["kappa"], sub["CSPR"], marker=MARKERS[p],
                color=COLORS[p], label=p, markersize=5, linewidth=1.4)
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel("Critical-service preservation ratio (CSPR)")
    ax.set_title("Critical-service preservation vs. surge")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5b_cspr_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 5c -- GDI vs kappa, 3 policies
#   X: kappa      Y: GDI      lines: policy
# ------------------------------------------------------------------------
def plot_fig5c_gdi_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P2", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        ax.plot(sub["kappa"], sub["GDI"], marker=MARKERS[p],
                color=COLORS[p], label=p, markersize=5, linewidth=1.4)
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel("Graceful-degradation index (GDI)")
    ax.set_title("Graceful degradation vs. surge")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5c_gdi_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 5d -- Admission ratio a3 vs kappa, P3 only (self-throttling curve)
#   X: kappa      Y: a3      (P3 only -- P1/P2 always keep a3=1 by design)
# ------------------------------------------------------------------------
def plot_fig5d_a3_vs_kappa(df):
    fig, ax = plt.subplots()
    sub = df[(df["policy"] == "P3") & df["feasible"]].sort_values("kappa")
    ax.step(sub["kappa"], sub["a3"], where="post", color=COLORS["P3"],
            marker="s", markersize=5, linewidth=1.6, label="P3 (proposed)")

    # P1/P2 always admit a3=1 whenever feasible, for reference
    for p in ["P1", "P2"]:
        sub_p = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        if len(sub_p):
            ax.scatter(sub_p["kappa"], sub_p["a3"], color=COLORS[p],
                       marker=MARKERS[p], s=25, alpha=0.6,
                       label=f"{p} (no throttling)")

    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel(r"Class-3 admission ratio $a_3$")
    ax.set_title("Self-throttling of Class 3 under P3")
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5d_a3_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 7a -- Class-1 PAoI (analytical surrogate) vs kappa, P1 vs P3
#   X: kappa      Y: PAoI1_s      lines: policy in {P1, P3}
# ------------------------------------------------------------------------
def plot_fig7a_paoi_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        ax.plot(sub["kappa"], sub["PAoI1_s"], marker=MARKERS[p],
                color=COLORS[p], label=p, markersize=5, linewidth=1.4)
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel(r"Class-1 peak AoI, $\bar{A}_1^{\mathrm{p}}$ [s]")
    ax.set_title("Class-1 peak AoI vs. surge: P1 vs P3")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig7a_paoi_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 7b -- Simulated time-average AoI vs kappa, P1 vs P3, with the
#              PAoI surrogate overlaid as dashed lines for comparison
#   X: kappa      Y: AoI_avg1_sim_s (solid), PAoI1_s (dashed)  lines: policy
# ------------------------------------------------------------------------
def plot_fig7b_avg_aoi_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa")
        ax.plot(sub["kappa"], sub["AoI_avg1_sim_s"], marker=MARKERS[p],
                color=COLORS[p], label=f"{p}, simulated avg AoI",
                markersize=5, linewidth=1.6)
        ax.plot(sub["kappa"], sub["PAoI1_s"], linestyle="--",
                color=COLORS[p], alpha=0.55, linewidth=1.2,
                label=f"{p}, PAoI surrogate")
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel("Class-1 age of information [s]")
    ax.set_title("Simulated average AoI vs. PAoI surrogate: P1 vs P3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig7b_avg_aoi_sim_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------------
# Figure 7c -- Gap between the PAoI surrogate and the simulated average AoI
#              (PAoI1_s - AoI_avg1_sim_s), P1 vs P3. Positive = surrogate
#              is conservative (>= simulated), consistent with Sec. VI.
#   X: kappa      Y: PAoI1_s - AoI_avg1_sim_s      lines: policy
# ------------------------------------------------------------------------
def plot_fig7c_surrogate_gap_vs_kappa(df):
    fig, ax = plt.subplots()
    for p in ["P1", "P3"]:
        sub = df[(df["policy"] == p) & df["feasible"]].sort_values("kappa").copy()
        sub["gap"] = sub["PAoI1_s"] - sub["AoI_avg1_sim_s"]
        ax.plot(sub["kappa"], sub["gap"], marker=MARKERS[p],
                color=COLORS[p], label=p, markersize=5, linewidth=1.4)
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Surge factor $\kappa$")
    ax.set_ylabel(r"PAoI surrogate $-$ simulated avg AoI [s]")
    ax.set_title("Conservativeness of the PAoI surrogate vs. surge")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig7c_surrogate_gap_vs_kappa.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    df = _load_exp5()
    print(f"[info] loaded {len(df)} rows from exp5 CSV "
          f"({df['feasible'].sum()} feasible)")

    # Figure 5
    plot_fig5a_delay_vs_kappa(df)
    plot_fig5b_cspr_vs_kappa(df)
    plot_fig5c_gdi_vs_kappa(df)
    plot_fig5d_a3_vs_kappa(df)

    # Figure 7
    plot_fig7a_paoi_vs_kappa(df)
    plot_fig7b_avg_aoi_vs_kappa(df)
    plot_fig7c_surrogate_gap_vs_kappa(df)

    print("\nAll Experiment 5 / 7 figures written to:", FIG_DIR)