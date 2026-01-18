import json
from pathlib import Path

import pandas as pd


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results_table(results_df, output_dir):
    """Save the main results table (price, duration, convexity) as CSV."""
    out = ensure_dir(output_dir)
    csv_path = out / "results_summary.csv"
    results_df.to_csv(csv_path, index=False)
    return csv_path


def save_calibration_params(calib_params, output_dir):
    """Save calibrated parameters as JSON for auditability."""
    out = ensure_dir(output_dir)
    path = out / "calibration_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib_params, f, indent=2, sort_keys=True, default=float)
    return path


def save_config_snapshot(cfg, output_dir):
    """Persist a subset of config fields as JSON (reproducibility)."""
    out = ensure_dir(output_dir)
    path = out / "config_snapshot.json"
    d = {}
    for k, v in cfg.__dict__.items():
        # Skip non-serializable objects.
        if k == "val_date":
            d[k] = str(v)
        elif isinstance(v, (int, float, str, bool)):
            d[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, sort_keys=True)
    return path


def maybe_plot_results(results_df, output_dir):
    """Create a few simple charts summarizing the comparison.

    Produces:
    - price_bar.png
    - duration_bar.png
    - convexity_bar.png

    If matplotlib is not available, this function does nothing.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out = ensure_dir(Path(output_dir) / "figures")
    paths = []

    def _bar(metric, filename, ylabel):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(results_df["method"], results_df[metric])
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(results_df["method"], rotation=45, ha="right")
        fig.tight_layout()
        p = out / filename
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)

    _bar("price", "price_bar.png", "Clean price")
    _bar("duration", "duration_bar.png", "Effective duration")
    _bar("convexity", "convexity_bar.png", "Effective convexity")

    return paths


def maybe_plot_curve_from_csv(curve_csv_path, output_dir):
    """Plot the input curve if the CSV is available.

    Expected columns (flexible): a date column and either discount factor or rate.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    curve_csv_path = Path(curve_csv_path)
    if not curve_csv_path.exists():
        return None

    df = pd.read_csv(curve_csv_path)
    # heuristic column discovery
    col_date = next((c for c in df.columns if "data" in c.lower() or "vertice" in c.lower() or "date" in c.lower()), None)
    col_df = next((c for c in df.columns if "fator" in c.lower() or "desconto" in c.lower() or "df" == c.lower()), None)
    col_rate = next((c for c in df.columns if "taxa" in c.lower() or "rate" in c.lower() or "zero" in c.lower()), None)

    if col_date is None:
        return None

    df[col_date] = pd.to_datetime(df[col_date])
    df = df.sort_values(col_date)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if col_df is not None:
        ax.plot(df[col_date], df[col_df])
        ax.set_ylabel("Discount factor")
    elif col_rate is not None:
        ax.plot(df[col_date], df[col_rate])
        ax.set_ylabel("Zero rate")
    else:
        return None

    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = ensure_dir(Path(output_dir) / "figures")
    p = out / "curve.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def maybe_plot_surface_from_csv(surface_csv_path, output_dir, title="surface"):
    """Plot a generic volatility surface CSV as a heatmap."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    surface_csv_path = Path(surface_csv_path)
    if not surface_csv_path.exists():
        return None

    df = pd.read_csv(surface_csv_path)
    if df.empty or len(df.columns) < 2:
        return None

    df = df.set_index(df.columns[0])
    values = df.values.astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out = ensure_dir(Path(output_dir) / "figures")
    p = out / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def save_hw_exercise_probabilities(state_cache, output_dir):
    """If available, save LSMC exercise probabilities by call date."""
    if not state_cache:
        return None

    # Two compatible formats are supported:
    # (A) Monolithic prototype: {"exercise_prob": {"YYYY-MM-DD": prob, ...}}
    # (B) Experimental format: {"exercise_probs_by_step": {step: prob}, "call_step_to_date": {step: "YYYY-MM-DD"}}
    if "exercise_prob" in state_cache and isinstance(state_cache.get("exercise_prob"), dict):
        rows = [{"call_date": str(d), "exercise_prob": float(p)} for d, p in state_cache["exercise_prob"].items()]
    else:
        probs = state_cache.get("exercise_probs_by_step")
        step_to_date = state_cache.get("call_step_to_date")
        if not probs or not step_to_date:
            return None

        rows = []
        for step, p in probs.items():
            rows.append({"call_date": step_to_date.get(step), "exercise_prob": float(p)})

    df = pd.DataFrame(rows).sort_values("call_date")

    out = ensure_dir(output_dir)
    path = out / "hw_lsmc_exercise_probabilities.csv"
    df.to_csv(path, index=False)

    # optional plot
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return path

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(df["call_date"], df["exercise_prob"])
    ax.set_ylabel("Exercise probability (risk-neutral)")
    ax.set_xlabel("Call date")
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_dir = ensure_dir(Path(output_dir) / "figures")
    fig_path = fig_dir / "hw_lsmc_exercise_probabilities.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return path


def save_dataframe(df, output_dir, filename):
    """Save a DataFrame to CSV inside ``output_dir``."""
    out = ensure_dir(output_dir)
    p = out / filename
    df.to_csv(p, index=False)
    return p


def maybe_plot_sensitivity(df, output_dir, x_col, title, xlabel, ylabel, filename_png):
    """Plot multiple lines from a wide sensitivity DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the x column (``x_col``) and one or more y columns.
    output_dir : str|Path
        Base output directory. The figure is saved under ``output_dir/figures``.
    x_col : str
        Name of the x-axis column.
    title, xlabel, ylabel : str
        Plot labels.
    filename_png : str
        Output filename (e.g. 'price_vs_oas.png').
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = df[x_col].values
    for col in df.columns:
        if col == x_col:
            continue
        ax.plot(x, df[col].values, marker="o", linewidth=1.5, label=str(col))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig_dir = ensure_dir(Path(output_dir) / "figures")
    p = fig_dir / filename_png
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p
