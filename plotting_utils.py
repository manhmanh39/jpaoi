# -*- coding: utf-8 -*-
"""
plotting_utils.py
================================================================================
Academic-style Matplotlib configuration shared by every figure generated for
the "Numerical Results and Simulation Validation" section of the paper.
================================================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {1: "#c0392b", 2: "#2980b9", 3: "#27ae60"}
CLASS_LABELS = {1: "Class 1 (C2/alarm)", 2: "Class 2 (SA telemetry)", 3: "Class 3 (best-effort)"}
CLASS_MARKERS = {1: "o", 2: "s", 3: "^"}

POLICY_COLORS = {"P1": "#7f8c8d", "P2": "#e67e22", "P3": "#8e44ad"}
POLICY_MARKERS = {"P1": "d", "P2": "s", "P3": "o"}
POLICY_LABELS = {
    "P1": "P1: fixed, no admission",
    "P2": "P2: resource adaptation",
    "P3": "P3: proposed service-aware orchestration",
}


def set_academic_style():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": True,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def new_fig(figsize=(5.2, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved figure] {path}")