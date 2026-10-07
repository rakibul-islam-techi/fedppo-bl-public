import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W, DBL_W = 3.5, 7.16

RC = {
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 4, "axes.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.75",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

DEADLINE = "#D55E00"
ABLATION = "#A6A6A6"

METHOD = {
    "FedPPO-QR":            dict(label="FedPPO-QR",                      color="#0072B2", ls="-",  marker="o"),
    "FedPPO-Mean":          dict(label="FedPPO-Mean",                    color="#E69F00", ls="--", marker="s"),
    "FedAvg-DDQN":          dict(label="FedAvg-DDQN",                    color="#009E73", ls=":",  marker="^"),
    "Centralized-QR":       dict(label="Non-Federated PPO-QR",           color="#56B4E9", ls="-.", marker="D"),
    "Heuristic":            dict(label="Static heuristic",               color="#CC79A7", ls="-.", marker="v"),
    "Heuristic-adaptive-n": dict(label="Adaptive-blocklength heuristic", color="#000000", ls="-",  marker="P"),
}

ABLATION_LABEL = {
    "block350": r"Fixed $n_k=350$",
    "block500": r"Fixed $n_k=500$",
    "block800": r"Fixed $n_k=800$",
    "block500_nohead": r"Fixed $n_k=500$, head removed",
    "rho05": r"Fixed $\rho_k=0.5$",
    "local": "Independent local learning",
    "local_shared": "Local learning, shared initialization",
    "agg_median": "Median aggregation",
    "agg_trimmed": "Trimmed-mean aggregation",
    "scale10": r"Federated with $B=10$",
}


def apply():
    plt.rcParams.update(RC)
    return plt
