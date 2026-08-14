# ------------------------------------------------------------------------
# Standalone figure generator functions for Experiment 3
# ------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import set_academic_style, CLASS_COLORS, CLASS_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH_EXP3 = os.path.join(BASE_DIR, "csv_tables", "exp3_alpha_sweep_waiting_delay.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_waiting_vs_alpha_uav(df):
    """Generates waiting_vs_alpha_uav.pdf: Class-wise UAV waiting time vs alpha."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    
    x = df["alpha"]
    ax.plot(x, df["W1_U_s"], marker='o', color=CLASS_COLORS[1], label=f"{CLASS_LABELS[1]}")
    ax.plot(x, df["W2_U_s"], marker='s', color=CLASS_COLORS[2], label=f"{CLASS_LABELS[2]}")
    ax.plot(x, df["W3_U_s"], marker='^', color=CLASS_COLORS[3], label=f"{CLASS_LABELS[3]}")

    ax.set_xlabel(r"Half-duplex split parameter $\alpha$")
    ax.set_ylabel(r"UAV waiting time $W_i^U$ [s]")
    ax.set_title(r"Class-wise UAV waiting time vs. $\alpha$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "waiting_vs_alpha_uav.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")

def plot_delay_vs_alpha_e2e(df):
    """Generates delay_vs_alpha_e2e.pdf: End-to-end delay vs alpha."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    
    x = df["alpha"]
    ax.plot(x, df["T1_E2E_s"], marker='o', color=CLASS_COLORS[1], label=f"{CLASS_LABELS[1]}")
    ax.plot(x, df["T2_E2E_s"], marker='s', color=CLASS_COLORS[2], label=f"{CLASS_LABELS[2]}")
    ax.plot(x, df["T3_E2E_s"], marker='^', color=CLASS_COLORS[3], label=f"{CLASS_LABELS[3]}")

    ax.set_xlabel(r"Half-duplex split parameter $\alpha$")
    ax.set_ylabel("Mean end-to-end delay [s]")
    ax.set_title(r"End-to-end delay vs. $\alpha$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "delay_vs_alpha_e2e.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")

if __name__ == "__main__":
    set_academic_style()
    if os.path.isfile(CSV_PATH_EXP3):
        df = pd.read_csv(CSV_PATH_EXP3)
        plot_waiting_vs_alpha_uav(df)
        plot_delay_vs_alpha_e2e(df)
        print("All Experiment 3 figures generated successfully!")
    else:
        print(f"[error] Could not find {CSV_PATH_EXP3}. Run experiment3_alpha_sweep() first.")