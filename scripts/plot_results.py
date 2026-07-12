import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "model",
    "kv_cache_type",
    "context_length",
    "peak_memory_mb",
    "latency_ms_per_token",
    "throughput_tokens_per_s",
    "perplexity",
    "status",
]


def load_csv_files(results_dir: Path, input_file: str | None = None) -> pd.DataFrame:
    if input_file:
        csv_files = [Path(input_file)]
    else:
        csv_files = sorted(results_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    numeric_cols = [
        "context_length",
        "peak_memory_mb",
        "latency_ms_per_token",
        "throughput_tokens_per_s",
        "perplexity",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_compiled_outputs(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    compiled_path = results_dir / "all_results_compiled.csv"
    df.to_csv(compiled_path, index=False, encoding="utf-8")

    summary = (
        df.dropna(subset=["peak_memory_mb"])
        .groupby(["model", "kv_cache_type", "context_length"], as_index=False)
        .agg(
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
            peak_memory_mb_std=("peak_memory_mb", "std"),
            latency_ms_per_token_mean=("latency_ms_per_token", "mean"),
            latency_ms_per_token_std=("latency_ms_per_token", "std"),
            throughput_tokens_per_s_mean=("throughput_tokens_per_s", "mean"),
            throughput_tokens_per_s_std=("throughput_tokens_per_s", "std"),
            perplexity_mean=("perplexity", "mean"),
            perplexity_std=("perplexity", "std"),
        )
    )

    summary_path = results_dir / "all_results_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    return summary


def plot_metric_by_context(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    plot_df = (
        df.dropna(subset=[metric])
        .groupby(["kv_cache_type", "context_length"], as_index=False)[metric]
        .mean()
        .sort_values(["kv_cache_type", "context_length"])
    )

    plt.figure(figsize=(10, 6))

    for method, group in plot_df.groupby("kv_cache_type"):
        plt.plot(
            group["context_length"],
            group[metric],
            marker="o",
            label=method,
        )

    plt.title(title)
    plt.xlabel("Context Length (tokens)")
    plt.ylabel(ylabel)
    plt.legend(title="KV Cache Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pareto(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = (
        df.dropna(subset=["peak_memory_mb", "perplexity"])
        .groupby(["kv_cache_type"], as_index=False)
        .agg(
            peak_memory_mb=("peak_memory_mb", "mean"),
            perplexity=("perplexity", "mean"),
        )
    )

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
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark CSV files and generate analysis plots."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing benchmark CSV files.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional single CSV file to analyze, e.g. results/template_log.csv.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv_files(results_dir, args.input_file)
    df = clean_data(df)

    save_compiled_outputs(df, results_dir)

    valid_df = df[df["status"].astype(str).str.contains("OK", na=False)].copy()

    plot_metric_by_context(
        valid_df,
        plots_dir / "vram_vs_context.png",
        "peak_memory_mb",
        "Peak VRAM (MB)",
        "Peak VRAM vs Context Length",
    )

    plot_metric_by_context(
        valid_df,
        plots_dir / "latency_vs_context.png",
        "latency_ms_per_token",
        "Latency (ms/token)",
        "Latency vs Context Length",
    )

    plot_metric_by_context(
        valid_df,
        plots_dir / "throughput_vs_context.png",
        "throughput_tokens_per_s",
        "Throughput (tokens/s)",
        "Throughput vs Context Length",
    )

    plot_pareto(
        valid_df,
        plots_dir / "pareto_ppl_vs_vram.png",
    )

    print("Done.")
    print(f"Compiled CSV: {results_dir / 'all_results_compiled.csv'}")
    print(f"Summary CSV:  {results_dir / 'all_results_summary.csv'}")
    print(f"Plots folder: {plots_dir}")


if __name__ == "__main__":
    main()
