import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 7,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.labelcolor": "black",
    "axes.unicode_minus": False,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "figure.titlesize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pgf.rcfonts": False,
})

PCB_COLOR = "#7A7A7A"
STUDY_DRUG_COLORS = {"DMT": "#D62728", "LSD": "#1F77B4"}
STUDY_ORDER = ["DMT", "LSD"]
HEMISPHERE_ORDER = ["left", "right"]
CONDITION_ORDER = ["PCB", "Drug"]


def paired_palette(study):
    return {"PCB": PCB_COLOR, "Drug": STUDY_DRUG_COLORS[study]}


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.labelcolor": "black",
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.titleweight": "bold",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "figure.titlesize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pgf.rcfonts": False,
    })
