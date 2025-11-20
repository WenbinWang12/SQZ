import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def aggregate_runs(root_results_dir, optimizers):
    """
    读取日志文件。
    假设结构: root_results_dir/<opt>/<opt>.csv
    """
    dfs = {}
    root = Path(root_results_dir)
    for opt in optimizers:
        # 兼容两种文件名模式：opt.csv 或 log.csv，根据上一轮代码是 opt.csv
        p = root / opt / f"{opt}.csv"
        if not p.exists():
            print(f"Warning: log not found for {opt}: {p}")
            continue
        try:
            df = pd.read_csv(p)
            # 确保数值类型正确
            numeric_cols = ['loss', 'tokens', 'wall_seconds', 'step', 'grad_norm', 'step_time']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs[opt] = df
        except Exception as e:
            print(f"Error reading {p}: {e}")
    return dfs

def plot_metrics_over_tokens(dfs, out_dir):
    """
    绘制 Loss, Grad Norm, Wall Clock 随 Tokens 的变化
    """
    out_dir = Path(out_dir)
    
    # 1. Tokens vs Loss
    plt.figure(figsize=(8, 5))
    for name, df in dfs.items():
        plt.plot(df['tokens'], df['loss'], label=name, alpha=0.8, linewidth=1.5)
    plt.xlabel('Tokens Processed')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Tokens')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'curve_loss_tokens.png', dpi=300)
    plt.close()

    # 2. Tokens vs Grad Norm (验证稳定性)
    plt.figure(figsize=(8, 5))
    for name, df in dfs.items():
        if 'grad_norm' in df.columns:
            # 使用 rolling mean 平滑一下，否则波动太大看不清
            smooth_norm = df['grad_norm'].rolling(window=5, min_periods=1).mean()
            plt.plot(df['tokens'], smooth_norm, label=name, alpha=0.7, linewidth=1)
    plt.xlabel('Tokens Processed')
    plt.ylabel('Gradient Norm (Smoothed)')
    plt.title('Gradient Norm Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'curve_grad_norm.png', dpi=300)
    plt.close()

def plot_efficiency_comparison(dfs, out_dir):
    """
    绘制步时 (Step Time) 和 总时间对比，验证 A-LS-Muon 的开销
    """
    out_dir = Path(out_dir)
    
    # 1. Average Step Time Bar Chart
    means = []
    names = []
    errs = []
    
    for name, df in dfs.items():
        if 'step_time' in df.columns:
            # 去掉前 10 步 warm-up
            valid_steps = df['step_time'].iloc[10:] if len(df) > 10 else df['step_time']
            means.append(valid_steps.mean() * 1000) # ms
            errs.append(valid_steps.std() * 1000)
            names.append(name)
    
    if names:
        plt.figure(figsize=(6, 4))
        plt.bar(names, means, yerr=errs, capsize=5, alpha=0.7, color=['tab:blue', 'tab:orange', 'tab:green'])
        plt.ylabel('Time per Step (ms)')
        plt.title('Computational Overhead Comparison')
        for i, v in enumerate(means):
            plt.text(i, v + 5, f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(out_dir / 'bar_step_time.png', dpi=300)
        plt.close()

def pareto_plot(dfs, out_path, target_loss=None):
    """
    修正后的 Pareto 图：
    X轴: 达到 target_loss 消耗的 Tokens
    Y轴: 达到 target_loss 消耗的 Wall-clock Time (而非总时间)
    """
    xs = []
    ys = []
    names = []
    
    plt.figure(figsize=(7, 5))
    
    for name, df in dfs.items():
        # 找到第一个小于等于 target_loss 的行
        if target_loss is not None:
            subset = df[df['loss'] <= target_loss]
            if not subset.empty:
                # 成功收敛
                row = subset.iloc[0]
                tok = row['tokens']
                time_sec = row['wall_seconds']
                marker = 'o'
            else:
                # 未收敛，使用最终状态 (用空心点表示)
                row = df.iloc[-1]
                tok = row['tokens']
                time_sec = row['wall_seconds']
                marker = 'x'
                name += " (Not Conv.)"
        else:
            # 如果没设阈值，就对比最终 Loss vs 时间
            # 但通常 Pareto 是 "资源 vs 效果"，这里我们假设对比最终点
            row = df.iloc[-1]
            tok = row['loss'] # 注意：这里改成了 Loss 作为 X 轴
            time_sec = row['wall_seconds']
            marker = 'o'

        xs.append(tok)
        ys.append(time_sec)
        names.append(name)
        plt.scatter(tok, time_sec, marker=marker, s=100, label=name)

    # 标注文字
    for i, txt in enumerate(names):
        plt.annotate(txt, (xs[i], ys[i]), xytext=(5, 5), textcoords='offset points')

    if target_loss is not None:
        plt.xlabel(f'Tokens to reach Loss < {target_loss}')
        plt.ylabel('Wall-clock Time (seconds)')
        plt.title(f'Pareto Efficiency (Target Loss: {target_loss})')
    else:
        plt.xlabel('Final Loss (Lower is better)')
        plt.ylabel('Total Training Time (s)')
        plt.title('Time vs Accuracy Trade-off')
        plt.gca().invert_xaxis() # Loss 越小越好，所以在右边

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_selection_heatmap_smart(selection_csv, out_path):
    """
    智能排序的 Heatmap：尝试按层级 ID (0, 1, 2...) 排序，而非频率。
    """
    if not Path(selection_csv).exists():
        return

    df = pd.read_csv(selection_csv)
    
    # 提取层号用于排序 helper function
    def get_layer_idx(name):
        # 匹配 h.0, layers.0, blocks.0 等常见模式
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return 9999 # 放最后

    # 添加辅助列用于排序
    df['layer_idx'] = df['param_name'].apply(get_layer_idx)
    
    # 先按层号排，再按参数名排
    df = df.sort_values(by=['layer_idx', 'param_name'], ascending=[True, True])
    
    # 绘图
    # 如果参数太多，只显示频率 > 0 的
    df = df[df['frequency'] > 0.001]
    
    plt.figure(figsize=(10, max(6, 0.25 * len(df))))
    plt.barh(df['param_name'], df['frequency'], color='teal')
    plt.xlabel('Selection Frequency')
    plt.ylabel('Layer / Parameter')
    plt.title('Adaptive Selection Distribution (Sorted by Depth)')
    plt.gca().invert_yaxis() # 让第一层在最上面
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def make_kpi_table(dfs, out_path_csv):
    rows = []
    for name, df in dfs.items():
        # 计算平均 step time
        avg_step_ms = df['step_time'].mean() * 1000 if 'step_time' in df.columns else 0
        
        row = {
            'optimizer': name,
            'tokens_processed': int(df['tokens'].iloc[-1]),
            'final_loss': round(float(df['loss'].iloc[-1]), 4),
            'total_time_sec': round(float(df['wall_seconds'].iloc[-1]), 2),
            'avg_step_time_ms': round(avg_step_ms, 1),
            'peak_grad_norm': round(float(df['grad_norm'].max()), 2) if 'grad_norm' in df.columns else 0
        }
        rows.append(row)
    
    out_df = pd.DataFrame(rows).set_index('optimizer')
    print("=== Experiment Summary ===")
    print(out_df)
    out_df.to_csv(out_path_csv)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/gpt2_125m_als_muon_v1')
    parser.add_argument('--target_loss', type=float, default=None, help="Loss threshold for Pareto plot")
    args = parser.parse_args()

    optimizers = ["AdamW", "FullMuon", "A-LS-Muon"]
    
    print(f"Analyzing results from: {args.results_dir}")
    dfs = aggregate_runs(args.results_dir, optimizers)
    
    if not dfs:
        print("No data found. Run training first.")
        exit()

    out_dir = Path(args.results_dir) / "plots"
    out_dir.mkdir(exist_ok=True)

    # 1. Basic Curves
    plot_metrics_over_tokens(dfs, out_dir)
    
    # 2. Efficiency Bars
    plot_efficiency_comparison(dfs, out_dir)
    
    # 3. Pareto Plot
    # 如果用户没指定 target_loss，我们尝试自动定一个（比如取三者中最差的那个 loss 作为基准，或者手动指定）
    # 这里简单逻辑：如果有 target_loss 参数就用，否则画 Final Loss vs Time
    pareto_plot(dfs, out_dir / 'pareto_efficiency.png', target_loss=args.target_loss)

    # 4. Heatmap (只有 A-LS-Muon 有)
    heatmap_csv = Path(args.results_dir) / "A-LS-Muon" / "selection_heatmap.csv"
    if heatmap_csv.exists():
        plot_selection_heatmap_smart(heatmap_csv, out_dir / 'heatmap_layers.png')

    # 5. Table
    make_kpi_table(dfs, out_dir / 'summary_metrics.csv')
    
    print(f"Analysis complete. Plots saved to {out_dir}")