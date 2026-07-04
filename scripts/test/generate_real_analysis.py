"""
Script tạo file Jupyter Notebook phân tích kết quả Benchmark THỰC TẾ.
Đọc file ../../results/template_log_real_run.csv và vẽ biểu đồ so sánh.

Cách chạy:
    python scripts/test/generate_real_analysis.py

Kết quả: ../../results/real_benchmark_analysis.ipynb
"""

import json


def main():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phân Tích Kết Quả Benchmark THỰC TẾ trên GPU\n",
                    "\n",
                    "Notebook này phân tích dữ liệu đo đạc **thật** từ GPU Cloud (RunPod/Vast.ai),\n",
                    "so sánh hiệu năng của 5 phương pháp KV Cache (FP16, FP8, HQQ, PolarQuant, TurboQuant)\n",
                    "trên 5 mô hình LLM tiếng Việt ở 3 mốc ngữ cảnh (4K, 8K, 16K).\n",
                    "\n",
                    "**Nguồn dữ liệu:** `../../results/template_log_real_run.csv`\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Công Thức Đo Lường\n",
                    "\n",
                    "### A. Peak VRAM (MB)\n",
                    "Được đo bằng **2 nguồn song song** và lấy giá trị lớn nhất:\n",
                    "- `pynvml`: Luồng nền (background thread) gọi `nvmlDeviceGetMemoryInfo()` mỗi 50ms\n",
                    "- `PyTorch`: `torch.cuda.max_memory_allocated()`\n",
                    "\n",
                    "$$\\text{Peak VRAM} = \\max(\\text{pynvml\\_peak}, \\text{torch\\_peak})$$\n",
                    "\n",
                    "### B. Latency (ms/token)\n",
                    "Thời gian trung bình để sinh ra 1 token trong pha Decode:\n",
                    "\n",
                    "$$\\text{Latency} = \\frac{\\text{Total Time (s)}}{\\text{Total Generated Tokens}} \\times 1000$$\n",
                    "\n",
                    "### C. Throughput (tokens/s)\n",
                    "Số token sinh ra được trong 1 giây (nghịch đảo của Latency):\n",
                    "\n",
                    "$$\\text{Throughput} = \\frac{\\text{Total Generated Tokens}}{\\text{Total Time (s)}}$$\n",
                    "\n",
                    "### D. Perplexity (PPL)\n",
                    "Đo chất lượng ngôn ngữ — mức độ bối rối của mô hình khi đoán từ tiếp theo:\n",
                    "\n",
                    "$$\\text{PPL} = \\exp\\left(-\\frac{1}{N}\\sum_{i=1}^{N}\\log P(x_i | x_{<i})\\right)$$\n",
                    "\n",
                    "PPL càng thấp = mô hình càng thông minh."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import numpy as np\n",
                    "\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "plt.rcParams.update({'figure.dpi': 120, 'font.size': 12})\n",
                    "\n",
                    "# Nap du lieu thuc te\n",
                    "df = pd.read_csv('../../results/template_log_real_run.csv')\n",
                    "df['peak_memory_mb'] = pd.to_numeric(df['peak_memory_mb'], errors='coerce')\n",
                    "df['latency_ms_per_token'] = pd.to_numeric(df['latency_ms_per_token'], errors='coerce')\n",
                    "df['throughput_tokens_per_s'] = pd.to_numeric(df['throughput_tokens_per_s'], errors='coerce')\n",
                    "df['perplexity'] = pd.to_numeric(df['perplexity'], errors='coerce')\n",
                    "\n",
                    "print(f'Tong so dong du lieu: {len(df)}')\n",
                    "print(f'So dong thanh cong (khong OOM): {len(df.dropna(subset=[\"peak_memory_mb\"]))}')\n",
                    "display(df.head(10))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. So sánh Peak VRAM theo từng phương pháp nén (Context 16K)\n",
                    "Biểu đồ cột nhóm thể hiện mức ngốn VRAM của 5 mô hình ở mốc ngữ cảnh cực đại."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_16k = df[df['context_length'] == 16000].dropna(subset=['peak_memory_mb'])\n",
                    "\n",
                    "if len(df_16k) > 0:\n",
                    "    plt.figure(figsize=(14, 7))\n",
                    "    sns.barplot(x='model', y='peak_memory_mb', hue='kv_cache_type', data=df_16k)\n",
                    "    plt.title('Peak VRAM - All Models @ Context 16K (Real GPU)', fontsize=16, fontweight='bold')\n",
                    "    plt.ylabel('Peak Memory (MB)')\n",
                    "    plt.xlabel('Model')\n",
                    "    plt.xticks(rotation=15)\n",
                    "    plt.legend(title='KV Cache Type')\n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('../../results/plots/real_vram_comparison_16k.png', dpi=150)\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Khong co du lieu context 16K. Kiem tra lai file CSV.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. VRAM Scaling theo Context Length\n",
                    "Đường biểu diễn sự gia tăng VRAM khi văn bản dài hơn — so sánh hệ số góc giữa FP16 và TurboQuant."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Chon model dau tien co du lieu de ve\n",
                    "available_models = df.dropna(subset=['peak_memory_mb'])['model'].unique()\n",
                    "if len(available_models) > 0:\n",
                    "    target_model = available_models[0]\n",
                    "    df_model = df[df['model'] == target_model].dropna(subset=['peak_memory_mb'])\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 6))\n",
                    "    sns.lineplot(x='context_length', y='peak_memory_mb', hue='kv_cache_type',\n",
                    "                 marker='o', linewidth=2.5, markersize=8, data=df_model)\n",
                    "    plt.title(f'VRAM Scaling with Context Length ({target_model.split(\"/\")[-1]})',\n",
                    "              fontsize=14, fontweight='bold')\n",
                    "    plt.ylabel('Peak Memory (MB)')\n",
                    "    plt.xlabel('Context Length (Tokens)')\n",
                    "    plt.xticks([4000, 8000, 16000])\n",
                    "    plt.grid(True, linestyle='--', alpha=0.7)\n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('../../results/plots/real_vram_scaling.png', dpi=150)\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Khong co du lieu hop le.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Memory vs Latency Trade-off (Pareto Frontier)\n",
                    "Mỗi chấm đại diện cho 1 phương pháp nén. Góc dưới-trái là vùng lý tưởng."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if len(available_models) > 0:\n",
                    "    df_pareto = df[(df['model'] == target_model) & (df['context_length'] == 16000)]\n",
                    "    df_pareto = df_pareto.dropna(subset=['peak_memory_mb', 'latency_ms_per_token'])\n",
                    "    \n",
                    "    if len(df_pareto) > 0:\n",
                    "        plt.figure(figsize=(10, 6))\n",
                    "        sns.scatterplot(x='peak_memory_mb', y='latency_ms_per_token', hue='kv_cache_type',\n",
                    "                        s=200, edgecolor='black', alpha=0.8, data=df_pareto)\n",
                    "        for i, row in df_pareto.iterrows():\n",
                    "            plt.text(row['peak_memory_mb'] + 100, row['latency_ms_per_token'],\n",
                    "                     row['kv_cache_type'], fontsize=10)\n",
                    "        plt.title(f'Memory vs Latency Trade-off ({target_model.split(\"/\")[-1]} @ 16K)',\n",
                    "                  fontsize=14, fontweight='bold')\n",
                    "        plt.xlabel('Peak Memory (MB) -> (Cang nho cang tot)')\n",
                    "        plt.ylabel('Latency (ms/token) -> (Cang nho cang tot)')\n",
                    "        plt.tight_layout()\n",
                    "        plt.savefig('../../results/plots/real_pareto_frontier.png', dpi=150)\n",
                    "        plt.show()\n",
                    "    else:\n",
                    "        print('Khong co du lieu 16K cho Pareto.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. So sánh Throughput giữa các phương pháp\n",
                    "Đo lượng token sinh ra mỗi giây — góc nhìn năng suất hệ thống."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_tp = df[df['context_length'] == 8000].dropna(subset=['throughput_tokens_per_s'])\n",
                    "\n",
                    "if len(df_tp) > 0:\n",
                    "    plt.figure(figsize=(14, 7))\n",
                    "    sns.barplot(x='model', y='throughput_tokens_per_s', hue='kv_cache_type', data=df_tp)\n",
                    "    plt.title('Throughput Comparison @ Context 8K (Real GPU)', fontsize=16, fontweight='bold')\n",
                    "    plt.ylabel('Throughput (tokens/s)')\n",
                    "    plt.xlabel('Model')\n",
                    "    plt.xticks(rotation=15)\n",
                    "    plt.legend(title='KV Cache Type')\n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('../../results/plots/real_throughput_8k.png', dpi=150)\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Khong co du lieu throughput.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Perplexity (PPL) - Chất lượng ngôn ngữ sau khi nén\n",
                    "Kiểm tra xem việc nén KV Cache có làm mô hình bị 'ngu đi' không."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_ppl = df.dropna(subset=['perplexity'])\n",
                    "\n",
                    "if len(df_ppl) > 0:\n",
                    "    plt.figure(figsize=(14, 7))\n",
                    "    sns.boxplot(x='kv_cache_type', y='perplexity', data=df_ppl,\n",
                    "                order=['FP16', 'FP8', 'HQQ', 'PolarQuant', 'TurboQuant'])\n",
                    "    plt.title('Perplexity Distribution by Compression Method (Real GPU)',\n",
                    "              fontsize=16, fontweight='bold')\n",
                    "    plt.ylabel('Perplexity (PPL) -> Cang thap cang tot')\n",
                    "    plt.xlabel('KV Cache Type')\n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('../../results/plots/real_perplexity_boxplot.png', dpi=150)\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Khong co du lieu PPL.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Bảng tổng hợp: % Tiết kiệm VRAM so với FP16 Baseline\n",
                    "\n",
                    "$$\\text{VRAM Reduction (\\%)} = \\left(1 - \\frac{\\text{Peak VRAM}_{\\text{Method}}}{\\text{Peak VRAM}_{\\text{FP16}}}\\right) \\times 100$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_valid = df.dropna(subset=['peak_memory_mb']).copy()\n",
                    "\n",
                    "# Lay gia tri FP16 lam moc\n",
                    "fp16_baseline = df_valid[df_valid['kv_cache_type'] == 'FP16'].set_index(\n",
                    "    ['model', 'context_length'])['peak_memory_mb']\n",
                    "\n",
                    "def calc_reduction(row):\n",
                    "    key = (row['model'], row['context_length'])\n",
                    "    if key in fp16_baseline.index:\n",
                    "        baseline = fp16_baseline[key]\n",
                    "        if baseline > 0:\n",
                    "            return round((1 - row['peak_memory_mb'] / baseline) * 100, 1)\n",
                    "    return None\n",
                    "\n",
                    "df_valid['vram_reduction_pct'] = df_valid.apply(calc_reduction, axis=1)\n",
                    "\n",
                    "summary = df_valid.groupby('kv_cache_type').agg({\n",
                    "    'peak_memory_mb': 'mean',\n",
                    "    'latency_ms_per_token': 'mean',\n",
                    "    'throughput_tokens_per_s': 'mean',\n",
                    "    'vram_reduction_pct': 'mean'\n",
                    "}).round(2)\n",
                    "\n",
                    "summary.columns = ['Avg Peak VRAM (MB)', 'Avg Latency (ms/tok)', 'Avg Throughput (tok/s)', 'Avg VRAM Reduction (%)']\n",
                    "display(summary)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path = "../../results/real_benchmark_analysis.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Da tao thanh cong: {output_path}")


if __name__ == "__main__":
    main()
