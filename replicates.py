"""Mean and standard error of the mean (SEM) from replicate measurements."""
import re
import numpy as np
import pandas as pd


def suggest_groups(columns, exclude=()):
    """Group column names that differ only by a trailing replicate index.
    Example: Cg_1, Cg_2, Cg_3 -> {'Cg': ['Cg_1', 'Cg_2', 'Cg_3']}."""
    pattern = re.compile(r"^(.*?)[\s_\-\.]*(?:rep|r)?[\s_\-\.]*\d+$", re.IGNORECASE)
    groups = {}
    for c in columns:
        if c in exclude:
            continue
        m = pattern.match(str(c))
        if m and m.group(1):
            groups.setdefault(m.group(1), []).append(c)
    return {g: cols for g, cols in groups.items() if len(cols) >= 2}


def _apply_policy(out, policy, sigma_floor):
    """Handle sigma == 0 or undefined (fewer than 2 replicates)."""
    bad = ~(out["sigma"] > 0)
    n_bad = int(bad.sum())
    if n_bad:
        if policy == "drop":
            out = out[~bad]
        elif policy == "floor":
            if sigma_floor is None or sigma_floor <= 0:
                raise ValueError("Provide a positive sigma floor.")
            out = out.copy()
            out.loc[bad, "sigma"] = sigma_floor
        else:
            raise ValueError("policy must be 'drop' or 'floor'")
    return out.reset_index(drop=True), n_bad


def replicate_table(df, x_col, rep_cols, policy="drop", sigma_floor=None):
    """Wide format: each replicate is a column. Returns (table, n_bad)."""
    reps = df[rep_cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": reps.mean(axis=1),
        "n": reps.count(axis=1),
    })
    out["sigma"] = reps.std(axis=1, ddof=1) / np.sqrt(out["n"])
    out = out.dropna(subset=["x", "y"])
    return _apply_policy(out, policy, sigma_floor)


def long_table(df, x_col, y_col, policy="drop", sigma_floor=None):
    """Long format: several rows per x value. Returns (table, n_bad)."""
    d = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    g = d.groupby(x_col)[y_col]
    out = pd.DataFrame({
        "x": g.mean().index.to_numpy(),
        "y": g.mean().to_numpy(),
        "n": g.count().to_numpy(),
    })
    out["sigma"] = g.std(ddof=1).to_numpy() / np.sqrt(out["n"])
    return _apply_policy(out, policy, sigma_floor)
