import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(df, funcs, metric, ylabel, filename_suffix):
    plt.figure(figsize=(8, 5))

    for func in funcs:
        sub = df[df['Function'] == func]
        plt.plot(sub['M'], sub[metric], marker='o', label=func)

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
