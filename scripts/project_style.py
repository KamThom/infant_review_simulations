import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

COLORS = {
    "REF": "#2F5597",
    "COMPARISON": "#D17A22",
    "gray_light": "#D9D9D9",
    "gray_mid": "#A6A6A6",
    "black": "#222222",
    "red": "#B23A48",
}

GROUP_LABELS = {
    "REF": "Reference group",
    "COMPARISON": "Comparison group",
}

POSITION_COLORS = {
    "held": "#7F7F7F",
    "supine": "#2F5597",
    "prone": "#D17A22",
    "sitting": "#70AD47",
}

POSITION_LABELS = {
    "held": "Held",
    "supine": "Supine",
    "prone": "Prone",
    "sitting": "Sitting",
}

def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
