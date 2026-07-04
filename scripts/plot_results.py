from __future__ import annotations

import argparse
import warnings
from pathlib import Path

RAW_EXCLUDED_SUFFIXES = ("_summary.csv", "_compiled.csv")
RAW_EXCLUDED_NAMES = {
    "template_log.csv",
    "all_results_compiled.csv",
    "all_results_summary.csv",
}


def is_raw_result_file(path: Path) -> bool:
    name = path.name
    if name in RAW_EXCLUDED_NAMES:
        return False
    if any(name.endswith(suffix) for suffix in RAW_EXCLUDED_SUFFIXES):
        return False
    return path.suffix.lower() == ".csv"


def discover_csv_files(results_dir: Path, input_file: str | None = None) -> list[Path]:
    if input_file:
        candidate = Path(input_file)
        return [candidate] if candidate.exists() and is_raw_result_file(candidate) else []
    return sorted(path for path in results_dir.glob("*.csv") if is_raw_result_file(path))


def load_csv_files(results_dir: Path, input_file: str | None = None) -> pd.DataFrame:
    import pandas as pd

    csv_files = discover_csv_files(results_dir, input_file)
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {results_dir}")

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ["model", "kv_cache_type", "context_length", "peak_memory_mb", "latency_ms_per_token", "throughput_tokens_per_s", "status"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_numeric_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    coerced = pd.to_numeric(df[col], errors="coerce")
    bad_count = coerced.isna().sum() - df[col].isna().sum()
    if bad_count > 0:
        warnings.warn(f"Column {col} has {bad_count} non-numeric values; coerced to NaN.", RuntimeWarning)
    df[col] = coerced


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd  # noqa: F401

    validate_columns(df)
    numeric_cols = [
        "context_length",
        "peak_memory_mb",
        "latency_ms_per_token",
        "throughput_tokens_per_s",
        "generated_tokens",
        "perplexity",
        "ppl_loss",
        "ppl_tokens",
        "repeated_ngram_ratio",
        "special_char_ratio",
        "output_length",
    ]
    for col in numeric_cols:
        clean_numeric_column(df, col)
    return df


def save_compiled_outputs(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    import pandas as pd

    compiled_path = results_dir / "all_results_compiled.csv"
    df.to_csv(compiled_path, index=False, encoding="utf-8")

    summary_cols = [
        "peak_memory_mb",
        "latency_ms_per_token",
        "throughput_tokens_per_s",
        "generated_tokens",
        "perplexity",
        "ppl_loss",
        "ppl_tokens",
        "output_length",
    ]
    available = [col for col in summary_cols if col in df.columns]
    group_cols = [col for col in ["model", "kv_cache_type", "context_length"] if col in df.columns]
    summary = df.dropna(subset=[col for col in ["peak_memory_mb"] if col in df.columns]).groupby(
        group_cols, as_index=False
    ).agg({col: ["mean", "std"] for col in available})
    if isinstance(summary.columns, pd.MultiIndex):
        summary.columns = [
            "_".join(filter(None, map(str, col))).rstrip("_")
            if isinstance(col, tuple)
            else col
            for col in summary.columns.to_flat_index()
        ]
    summary_path = results_dir / "all_results_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    return summary


def plot_metric_by_context(df: pd.DataFrame, output_path: Path, metric: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        warnings.warn(f"Skipping plot for missing metric: {metric}", RuntimeWarning)
        return

    plot_df = (
        df.dropna(subset=[metric])
        .groupby(["kv_cache_type", "context_length"], as_index=False)[metric]
        .mean()
        .sort_values(["kv_cache_type", "context_length"])
    )
    if plot_df.empty:
        warnings.warn(f"No valid data for plot: {metric}", RuntimeWarning)
        return

    plt.figure(figsize=(10, 6))
    for method, group in plot_df.groupby("kv_cache_type"):
        plt.plot(group["context_length"], group[metric], marker="o", label=method)
    plt.title(title)
    plt.xlabel("Context Length (tokens)")
    plt.ylabel(ylabel)
    plt.legend(title="KV Cache Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pareto(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if "perplexity" not in df.columns:
        warnings.warn("Skipping pareto plot because perplexity column is missing.", RuntimeWarning)
        return

    plot_df = (
        df.dropna(subset=["peak_memory_mb", "perplexity"])
        .groupby(["kv_cache_type"], as_index=False)
        .agg(
            peak_memory_mb=("peak_memory_mb", "mean"),
            perplexity=("perplexity", "mean"),
        )
    )
    if plot_df.empty:
        warnings.warn("No valid data for pareto plot.", RuntimeWarning)
        return

    plt.figure(figsize=(9, 6))
    for _, row in plot_df.iterrows():
        plt.scatter(row["peak_memory_mb"], row["perplexity"], s=90)
        plt.annotate(
            row["kv_cache_type"],
            (row["peak_memory_mb"], row["perplexity"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    plt.title("Pareto Trade-off: Perplexity vs Peak VRAM")
    plt.xlabel("Average Peak VRAM (MB)")
    plt.ylabel("Average Perplexity (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Aggregate benchmark CSV files and generate analysis plots.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--input-file", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv_files(results_dir, args.input_file)
    df = clean_data(df)
    save_compiled_outputs(df, results_dir)

    valid_df = df[df["status"].astype(str).str.contains("OK", na=False)].copy()

    plot_metric_by_context(valid_df, plots_dir / "vram_vs_context.png", "peak_memory_mb", "Peak VRAM (MB)", "Peak VRAM vs Context Length")
    plot_metric_by_context(valid_df, plots_dir / "latency_vs_context.png", "latency_ms_per_token", "Latency (ms/token)", "Latency vs Context Length")
    plot_metric_by_context(valid_df, plots_dir / "throughput_vs_context.png", "throughput_tokens_per_s", "Throughput (tokens/s)", "Throughput vs Context Length")
    plot_pareto(valid_df, plots_dir / "pareto_ppl_vs_vram.png")

    print("Done.")
    print(f"Compiled CSV: {results_dir / 'all_results_compiled.csv'}")
    print(f"Summary CSV:  {results_dir / 'all_results_summary.csv'}")
    print(f"Plots folder: {plots_dir}")


if __name__ == "__main__":
    main()
