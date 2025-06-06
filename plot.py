import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Define a consistent color mapping for functions
FUNCTION_COLORS = {
    'naive': '#6B8E9E',        # muted blue
    'mem_aliasing': '#D4A5A5',  # muted red
    'loop_unrolling_x1': '#9EBE9E',  # muted green
    'loop_unrolling_x3': '#B8A5D4',  # muted purple
    'cache_blocking': '#E6C8A5',     # muted orange
    'simd': '#A5A5A5'          # muted gray
}

def plot_metrics(df, funcs, metric, ylabel, filename_suffix):
    plt.figure(figsize=(8, 5))

    for func in funcs:
        sub = df[df['Function'] == func]
        color = FUNCTION_COLORS.get(func, 'gray')  # Use gray as fallback for unknown functions
        plt.plot(sub['M'], sub[metric], marker='o', label=func, color=color)

    plt.xlabel("Matrix Size (M=N=K)")
    plt.ylabel(ylabel)
    plt.title(f"GEMM {ylabel}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{metric.lower()}_{filename_suffix}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot GEMM benchmark results.")
    parser.add_argument("functions", nargs="*", help="GEMM function names to plot (default: all)")
    args = parser.parse_args()

    df = pd.read_csv("benchmark_results.csv")
    all_funcs = df['Function'].unique()

    if args.functions:
        selected_funcs = [f for f in args.functions if f in all_funcs]
        if not selected_funcs:
            print("None of the specified functions found in the data.")
            return
    else:
        selected_funcs = all_funcs

    suffix = "_".join(selected_funcs)
    plot_metrics(df, selected_funcs, metric="GFLOPS", ylabel="GFLOPS", filename_suffix=suffix)
    plot_metrics(df, selected_funcs, metric="TimeSec", ylabel="Execution Time (s)", filename_suffix=suffix)


if __name__ == "__main__":
    main()
