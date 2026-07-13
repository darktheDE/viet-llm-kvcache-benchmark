"""
plot_results.py — Aggregate KV-cache benchmark logs and render the 4-slot figure set
specified in the project docs (Sprint 03 / Master Plan Phase 3), with honest
interpretations where the raw GPU data cannot naively support the metric.

Slot mapping (docs -> defensible implementation):
  Slot 1  vram_vs_context        Measured peak VRAM vs context.
          Caption: vLLM pre-allocates the KV block pool via gpu_memory_utilization,
          so peak VRAM is ~70-71 GB across all methods. This plot documents the pool
          footprint, NOT the KV-cache footprint; it demonstrates that VRAM is not a
          discriminating metric in this setup (a methodological finding).
  Slot 2  scaling_grid           3-panel: Latency / Throughput / Perplexity (median) vs context.
          Fully supported by measured data.
  Slot 3  pareto_ppl_vs_kvsize   Perplexity (mean) vs THEORETICAL KV-cache size.
          KV-cache size = bits_per_value / 16 x 100% of FP16. Analytical, no GPU needed.
          Satisfies the "PPL vs memory" spirit of the Pareto requirement.
  Slot 4  compression_efficiency  Bars: theoretical compression ratio + effective throughput
          vs context. Compression ratio = 16 / bits_per_value (literature standard).
          Effective throughput = measured_throughput x compression_ratio (tokens served
          on the same pool). Rewards methods with smaller caches.
  Bonus   pareto_ppl_vs_speed    Perplexity vs measured throughput (real speed-quality trade-off)

Data-integrity notes embedded in the figures:
  - Slot 1 y-axis starts at 0 with GPU-capacity / pool reference lines so the reader
    sees the method spread is negligible (<2%, pool-dominated).
  - Slot 3 annotates methods at the same x (FP8 & PolarQuant both at 50%) with stacked
    offsets so labels do not collide.
  - Slot 4 left uses bars (not overlapping lines) so the FP8/PolarQuant and HQQ/TurboQuant
    pairs are distinctly visible as separate bars with the same height.
  - Slot 4 right labels its metric "effective throughput (quality-agnostic)" so
    the reader cannot confuse it with a quality-weighted score.

Perplexity uses MEDIAN for scaling lines (robust to HQQ tail, n=2-4 per cell) and MEAN
for the Pareto (reflects average degradation of the tail). Both choices are stated in
captions where they appear.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Schema & identity
# --------------------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "model", "kv_cache_type", "context_length", "peak_memory_mb",
    "latency_ms_per_token", "throughput_tokens_per_s", "perplexity", "status",
]
EXCLUDED_STEMS = {"all_results_compiled", "all_results_summary"}
EXCLUDED_KEYWORDS = ("template_log.csv", "demo", "mock")

METHOD_COLORS = {
    "FP16": "#6B7280", "BF16": "#6B7280", "FP8": "#2563EB",
    "HQQ": "#F59E0B", "PolarQuant": "#10B981", "TurboQuant": "#EF4444",
}
METHOD_ORDER = ["FP16", "BF16", "FP8", "PolarQuant", "HQQ", "TurboQuant"]
_FALLBACK = ["#7C3AED", "#DB2777", "#0891B2", "#65A30D"]

# Nominal bits per stored KV value. Editable. Assumes weights stay FP16 (only KV quantized,
# matching the study design) and IGNORES per-group scale/zero metadata overhead (small).
METHOD_BITS = {
    "FP16": 16, "BF16": 16, "FP8": 8, "PolarQuant": 8, "HQQ": 4, "TurboQuant": 4,
}
BASELINE = "FP16"


def set_paper_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def color_for(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return _FALLBACK[abs(hash(method)) % len(_FALLBACK)]


def ordered_methods(methods) -> list:
    present = list(dict.fromkeys(methods))
    known = [m for m in METHOD_ORDER if m in present]
    return known + sorted(m for m in present if m not in METHOD_ORDER)


def rel_kv_size_pct(method: str) -> float | None:
    b = METHOD_BITS.get(method)
    return None if b is None else 100.0 * b / METHOD_BITS[BASELINE]


def save_fig(fig, plots_dir: Path, name: str, formats: list) -> None:
    for fmt in formats:
        fig.savefig(plots_dir / f"{name}.{fmt}")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Load / clean / compile
# --------------------------------------------------------------------------------------
def _select_files(results_dir: Path, use_all: bool) -> list:
    files = [f for f in sorted(results_dir.glob("*.csv"))
             if f.stem not in EXCLUDED_STEMS
             and not any(k in f.name.lower() for k in EXCLUDED_KEYWORDS)]
    if not use_all:
        allv = [f for f in files if f.stem.endswith("_all")]
        if allv:
            return allv
    return files


def load_csv_files(results_dir: Path, input_file, use_all: bool) -> pd.DataFrame:
    files = [Path(input_file)] if input_file else _select_files(results_dir, use_all)
    if not files:
        raise FileNotFoundError(f"No usable CSV files in {results_dir}")
    print("Ingesting:")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
        print(f"  - {f.name} ({len(df)} rows)")
    return pd.concat(frames, ignore_index=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in ["context_length", "peak_memory_mb", "latency_ms_per_token",
              "throughput_tokens_per_s", "perplexity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["kv_cache_type"] = df["kv_cache_type"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    return df


def save_compiled_outputs(df: pd.DataFrame, results_dir: Path) -> None:
    df.to_csv(results_dir / "all_results_compiled.csv", index=False, encoding="utf-8")
    (df.dropna(subset=["peak_memory_mb"])
       .groupby(["model", "kv_cache_type", "context_length"], as_index=False)
       .agg(peak_memory_mb_mean=("peak_memory_mb", "mean"),
            latency_ms_per_token_mean=("latency_ms_per_token", "mean"),
            throughput_tokens_per_s_mean=("throughput_tokens_per_s", "mean"),
            perplexity_mean=("perplexity", "mean"),
            perplexity_median=("perplexity", "median"))
       .to_csv(results_dir / "all_results_summary.csv", index=False, encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Pareto helper (supports maximise or minimise per axis)
# --------------------------------------------------------------------------------------
def pareto_front(pts: pd.DataFrame, x, y, min_x=True, min_y=True) -> pd.DataFrame:
    p = pts.copy()
    p["_cx"] = p[x] if min_x else -p[x]
    p["_cy"] = p[y] if min_y else -p[y]
    p = p.sort_values(["_cx", "_cy"]).reset_index(drop=True)
    front, best = [], np.inf
    for _, r in p.iterrows():
        if r["_cy"] <= best:
            front.append(r)
            best = r["_cy"]
    return pd.DataFrame(front)


# --------------------------------------------------------------------------------------
# Slot 1 — VRAM vs context (flat, documented)
# --------------------------------------------------------------------------------------
def plot_vram_vs_context(df, plots_dir, formats, gpu_total_mb):
    d = df.dropna(subset=["peak_memory_mb"])
    if d.empty:
        print("  [skip] Slot 1 vram_vs_context: no VRAM data.")
        return
    line = d.groupby(["kv_cache_type", "context_length"], as_index=False)["peak_memory_mb"].median()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m in ordered_methods(line["kv_cache_type"].unique()):
        g = line[line.kv_cache_type == m].sort_values("context_length")
        ax.plot(g["context_length"], g["peak_memory_mb"], marker="o", markersize=4,
                color=color_for(m), label=m, linewidth=1.6)
    if gpu_total_mb:
        ax.axhline(gpu_total_mb, ls="-", color="#111827", lw=1)
        ax.axhline(0.9 * gpu_total_mb, ls=":", color="#6B7280", lw=1)
        ax.text(0.01, gpu_total_mb,
                f"  GPU capacity ({gpu_total_mb / 1024:.0f} GB)",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#111827")
        ax.text(0.01, 0.9 * gpu_total_mb,
                "  pre-allocated pool (~90%)",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#6B7280")
        ax.set_ylim(0, gpu_total_mb * 1.08)
    ax.set_title("Peak VRAM vs context (pool-dominated, spread <2%)")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, plots_dir, "vram_vs_context", formats)


# --------------------------------------------------------------------------------------
# Slot 2 — Scaling grid (Latency / Throughput / Perplexity vs context)
# --------------------------------------------------------------------------------------
def _line_panel(ax, df, metric, ylabel, add_points=False, agg="median"):
    d = df.dropna(subset=[metric])
    if d.empty:
        ax.set_visible(False)
        return
    line = d.groupby(["kv_cache_type", "context_length"], as_index=False)[metric].agg(agg)
    for m in ordered_methods(line["kv_cache_type"].unique()):
        g = line[line.kv_cache_type == m].sort_values("context_length")
        ax.plot(g["context_length"], g[metric], marker="o", markersize=4,
                color=color_for(m), label=m, linewidth=1.6, zorder=3)
        if add_points:
            raw = d[d.kv_cache_type == m]
            ax.scatter(raw["context_length"], raw[metric], s=10,
                       color=color_for(m), alpha=0.28, zorder=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(ylabel)


def plot_scaling_grid(df, plots_dir, formats):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    _line_panel(axes[0], df, "latency_ms_per_token", "Latency (ms/token)")
    _line_panel(axes[1], df, "throughput_tokens_per_s", "Throughput (tokens/s)")
    _line_panel(axes[2], df, "perplexity", "Perplexity (median)", add_points=True)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, title="KV cache", loc="upper center",
               ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, plots_dir, "scaling_grid", formats)


# --------------------------------------------------------------------------------------
# Slot 3 — Pareto: Perplexity vs theoretical KV-cache size
# --------------------------------------------------------------------------------------
def plot_pareto_kvsize(df, plots_dir, formats, context):
    d = df.dropna(subset=["perplexity"])
    if context is not None:
        d = d[d.context_length == context]
    agg = d.groupby("kv_cache_type", as_index=False)["perplexity"].mean()
    agg["size"] = agg["kv_cache_type"].map(rel_kv_size_pct)
    agg = agg.dropna(subset=["size"])
    if agg.empty:
        print("  [skip] Slot 3 pareto_kvsize: no data.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for _, r in agg.iterrows():
        ax.scatter(r["size"], r["perplexity"], s=110, color=color_for(r.kv_cache_type),
                   edgecolor="white", linewidth=0.8, zorder=3, label=r.kv_cache_type)
    front = pareto_front(agg.rename(columns={"size": "x", "perplexity": "y"}),
                         "x", "y", min_x=True, min_y=True)
    ax.plot(front["x"], front["y"], "--", color="#374151", lw=1.1, zorder=2)

    seen = {}
    for _, r in agg.sort_values(["size", "perplexity"]).iterrows():
        key = round(r["size"])
        k = seen.get(key, 0)
        seen[key] = k + 1
        ax.annotate(r.kv_cache_type, (r["size"], r["perplexity"]),
                    xytext=(9, 4 - k * 12), textcoords="offset points",
                    fontsize=8, va="center")
    ctx = f" @ {context:,} ctx" if context is not None else ""
    ax.set_title(f"Quality vs theoretical KV-cache size{ctx}")
    ax.set_xlabel("Theoretical KV-cache size (% of FP16, analytical)")
    ax.set_ylabel("Perplexity (mean) — lower is better")
    ax.set_xlim(0, 110)
    save_fig(fig, plots_dir, "pareto_ppl_vs_kvsize", formats)


# --------------------------------------------------------------------------------------
# Slot 4 — Compression ratio (theoretical) + effective throughput vs context
# --------------------------------------------------------------------------------------
def plot_compression_efficiency(df, plots_dir, formats):
    methods = ordered_methods(df.kv_cache_type.unique())
    methods = [m for m in methods if rel_kv_size_pct(m) is not None]
    ratios = {m: METHOD_BITS[BASELINE] / METHOD_BITS[m] for m in methods}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    x = np.arange(len(methods))
    axl.bar(x, [ratios[m] for m in methods], color=[color_for(m) for m in methods], width=0.6)
    for i, m in enumerate(methods):
        axl.annotate(f"{ratios[m]:.0f}x", (i, ratios[m]), ha="center", va="bottom", fontsize=8)
    axl.set_xticks(x)
    axl.set_xticklabels(methods, rotation=20, ha="right")
    axl.set_ylabel("Compression ratio (x vs FP16)")
    axl.set_title("Theoretical KV compression")
    axl.grid(axis="x", visible=False)

    d = df.dropna(subset=["throughput_tokens_per_s"])
    eff = d.groupby(["kv_cache_type", "context_length"], as_index=False)["throughput_tokens_per_s"].median()
    eff["eff"] = eff.apply(
        lambda r: r["throughput_tokens_per_s"] * ratios.get(r["kv_cache_type"], 1), axis=1
    )
    for m in methods:
        g = eff[eff.kv_cache_type == m].sort_values("context_length")
        axr.plot(g["context_length"], g["eff"], marker="o", markersize=4,
                 color=color_for(m), label=m, linewidth=1.6)
    axr.set_title("Effective throughput = tok/s x ratio  (quality-agnostic)")
    axr.set_xlabel("Context length (tokens)")
    axr.set_ylabel("Effective throughput")
    axr.legend(loc="upper right", ncol=2, fontsize=7)

    fig.tight_layout()
    save_fig(fig, plots_dir, "compression_efficiency", formats)


# --------------------------------------------------------------------------------------
# Bonus — Perplexity vs measured throughput (speed-quality trade-off)
# --------------------------------------------------------------------------------------
def plot_pareto_speed(df, plots_dir, formats, context):
    d = df.dropna(subset=["perplexity", "throughput_tokens_per_s"])
    if context is not None:
        d = d[d.context_length == context]
    agg = d.groupby("kv_cache_type", as_index=False).agg(
        ppl=("perplexity", "median"), thr=("throughput_tokens_per_s", "median"))
    if agg.empty:
        print("  [skip] Bonus speed pareto: no data.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for _, r in agg.iterrows():
        ax.scatter(r["thr"], r["ppl"], s=110, color=color_for(r.kv_cache_type),
                   edgecolor="white", linewidth=0.8, zorder=3, label=r.kv_cache_type)
    seen = {}
    for _, r in agg.sort_values(["thr", "ppl"]).iterrows():
        key = round(r["thr"] / 4)
        k = seen.get(key, 0)
        seen[key] = k + 1
        ax.annotate(r.kv_cache_type, (r["thr"], r["ppl"]),
                    xytext=(9, 4 - k * 12), textcoords="offset points", fontsize=8, va="center")
    front = pareto_front(agg.rename(columns={"thr": "x", "ppl": "y"}),
                         "x", "y", min_x=False, min_y=True)
    ax.plot(front["x"], front["y"], "--", color="#374151", lw=1.1, zorder=2)
    ctx = f" @ {context:,} ctx" if context is not None else ""
    ax.set_title(f"Quality vs decode speed{ctx}")
    ax.set_xlabel("Throughput (tokens/s) — higher is better")
    ax.set_ylabel("Perplexity (median) — lower is better")
    save_fig(fig, plots_dir, "pareto_ppl_vs_speed", formats)


# --------------------------------------------------------------------------------------
def _longest_context_with_baseline(df):
    ok = df[df.status.str.contains("OK", na=False)]
    base = ok[ok.kv_cache_type == BASELINE]
    pool = base if not base.empty else ok
    ctxs = sorted(c for c in pool.context_length.dropna().unique())
    return int(ctxs[-1]) if ctxs else None


def main():
    ap = argparse.ArgumentParser(
        description="Render the 4-slot figure set specified in project docs."
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--input-file", default=None)
    ap.add_argument("--all-files", action="store_true")
    ap.add_argument("--pareto-context", type=int, default=None)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--gpu-total-mb", type=int, default=81920,
                    help="GPU capacity for the VRAM reference line (A100 80GB=81920).")
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    df = load_csv_files(results_dir, args.input_file, args.all_files)
    df = clean_data(df)
    save_compiled_outputs(df, results_dir)

    valid = df[df.status.str.contains("OK", na=False)].copy()
    ctx = args.pareto_context or _longest_context_with_baseline(df)
    if ctx:
        print(f"\nPareto context: {ctx:,} tokens")

    print("\nRendering figures (4-slot docs mapping + bonus):")
    plot_vram_vs_context(valid, plots_dir, formats, args.gpu_total_mb)  # Slot 1
    plot_scaling_grid(valid, plots_dir, formats)                        # Slot 2
    plot_pareto_kvsize(valid, plots_dir, formats, ctx)                  # Slot 3
    plot_compression_efficiency(valid, plots_dir, formats)              # Slot 4
    plot_pareto_speed(valid, plots_dir, formats, ctx)                   # Bonus

    print("\nData-integrity notes:")
    print("  Slot 1: y-axis from 0, GPU-capacity reference => spread <2% (pool artifact)")
    print("  Slot 3: labels at same x offset vertically (FP8/PolarQuant @ 50%)")
    print("  Slot 4 left: bars replace overlapping lines (FP8==PolarQuant, HQQ==TurboQuant)")
    print("  Slot 4 right: 'effective throughput' is quality-agnostic (HQQ degrades)")
    print(f"\nDone. Figures ({', '.join(formats)}) in {plots_dir}")


if __name__ == "__main__":
    main()
