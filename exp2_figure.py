# ------------------------------------------------------------------------
# Standalone figure generator functions for Experiment 2 (Sec. XI-A / Sec. IV)
# Add these functions to your make_experiment2_figures.py or equivalent script.
# ------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import set_academic_style, CLASS_COLORS, CLASS_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH_EXP2 = os.path.join(BASE_DIR, "csv_tables", "exp2_snr_sweep_delay_bottleneck.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_delay_vs_snr_uh(df):
    """Generates delay_vs_snr_uh.pdf: Class-wise E2E delay versus bar_gamma_UH."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    
    x = df["bar_gamma_UH"]
    ax.plot(x, df["T1_E2E_s"], marker='o', color=CLASS_COLORS[1], label=f"{CLASS_LABELS[1]} (Class 1)")
    ax.plot(x, df["T2_E2E_s"], marker='s', color=CLASS_COLORS[2], label=f"{CLASS_LABELS[2]} (Class 2)")
    ax.plot(x, df["T3_E2E_s"], marker='^', color=CLASS_COLORS[3], label=f"{CLASS_LABELS[3]} (Class 3)")

    ax.set_xlabel(r"UAV--HAP Mean SNR $\bar{\gamma}_{UH}$ [dB or linear scale]")
    ax.set_ylabel("Mean end-to-end delay [s]")
    ax.set_title("Class-wise end-to-end mean delay vs. $\\bar{\\gamma}_{UH}$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "delay_vs_snr_uh.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")

def plot_bottleneck_vs_snr_uh(df):
    """Generates bottleneck_vs_snr_uh.pdf: Stage-wise aggregate utilization and D_rho versus bar_gamma_UH."""
    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))

    x = df["bar_gamma_UH"]
    
    # Left y-axis: Utilization
    line1 = ax1.plot(x, df["rho_U"], marker='o', color="#e74c3c", label=r"UAV utilization ($\rho^U$)")
    line2 = ax1.plot(x, df["rho_H"], marker='s', color="#3498db", label=r"HAP utilization ($\rho^H$)")
    ax1.set_xlabel(r"UAV--HAP Mean SNR $\bar{\gamma}_{UH}$")
    ax1.set_ylabel("Aggregate Utilization", color="black")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    # Right y-axis: Bottleneck indicator D_rho
    ax2 = ax1.twinx()
    line3 = ax2.plot(x, df["D_rho"], marker='^', color="#27ae60", linestyle="--", label=r"Bottleneck indicator ($D_\rho$)")
    ax2.set_ylabel(r"Bottleneck Indicator $D_\rho$", color="#27ae60")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9, frameon=True)

    # ax1.set_title("Stage-wise aggregate utilization and D_\rho$ vs. $\\bar{\\gamma}_{UH}$")
    ax1.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "bottleneck_vs_snr_uh.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved figure] {out_path}")

if __name__ == "__main__":
    set_academic_style()
    if os.path.isfile(CSV_PATH_EXP2):
        df = pd.read_csv(CSV_PATH_EXP2)
        plot_delay_vs_snr_uh(df)
        plot_bottleneck_vs_snr_uh(df)
        print("All Experiment 2 figures generated successfully!")
    else:
        print(f"[error] Could not find {CSV_PATH_EXP2}. Run experiment2_snr_sweep() first.")