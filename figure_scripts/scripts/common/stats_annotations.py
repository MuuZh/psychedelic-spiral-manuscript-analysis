import numpy as np
from scipy.stats import pearsonr, ttest_rel


def correlation_text(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return f"n={mask.sum()}"
    r, p = pearsonr(np.asarray(x)[mask], np.asarray(y)[mask])
    return f"r={r:.2f}\np={p:.2g}\nn={mask.sum()}"


def paired_test_text(data):
    wide = data.pivot(index="subject", columns="condition", values="value").dropna()
    if len(wide) < 2:
        return f"paired n={len(wide)}"
    _, p = ttest_rel(wide["Drug"], wide["PCB"])
    return f"paired n={len(wide)}\np={p:.2g}"


def paired_significance_text(data):
    wide = data.pivot(index="subject", columns="condition", values="value").dropna()
    if len(wide) < 2:
        return "NS"
    _, p = ttest_rel(wide["Drug"], wide["PCB"])
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"
