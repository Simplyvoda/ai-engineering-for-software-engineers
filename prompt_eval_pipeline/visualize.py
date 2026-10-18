import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# One fixed colour and marker per prompt, assigned in name order
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# The first line is solid and later ones dashed, so tied lines both stay visible
SERIES_DASHES = [(None, None), (6, 3), (2, 2), (8, 3, 2, 3), (4, 2), (1, 2), (10, 3), (3, 3)]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

def load_runs(runs_dir: str):
    """Load every saved run, oldest first."""
    runs = []
    for run_file in sorted(glob.glob(os.path.join(runs_dir, "run_*.json"))):
        with open(run_file) as f:
            runs.append(json.load(f))
    return runs

def run_label(run_id: str) -> str:
    """Turn a run id like 20261008_162729 into 'Oct 08\\n16:27'."""
    return datetime.strptime(run_id, "%Y%m%d_%H%M%S").strftime("%b %d\n%H:%M")

def spread_labels(values, min_gap):
    """Nudge label positions apart so equal values don't print on top of each other."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    positions = list(values)
    for prev, cur in zip(order, order[1:]):
        if positions[cur] - positions[prev] < min_gap:
            positions[cur] = positions[prev] + min_gap
    return positions

def plot_accuracy_over_time(runs, output_file: str, metric: str = "label_accuracy"):
    """Line chart: one line per prompt, one point per run."""
    prompt_names = sorted({s['prompt_name'] for run in runs for s in run['summaries']})
    if len(prompt_names) > len(SERIES_COLORS):
        raise ValueError(f"Too many prompts to chart ({len(prompt_names)}); max is {len(SERIES_COLORS)}")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    end_points = []
    for i, name in enumerate(prompt_names):
        xs, ys = [], []
        for x, run in enumerate(runs):
            for summary in run['summaries']:
                if summary['prompt_name'] == name:
                    xs.append(x)
                    ys.append(summary[metric])

        ax.plot(
            xs, ys,
            color=SERIES_COLORS[i],
            marker=SERIES_MARKERS[i],
            linewidth=2,
            dashes=SERIES_DASHES[i],
            markersize=8 + 3 * (len(prompt_names) - 1 - i),
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
            clip_on=False,
            zorder=3
        )
        end_points.append((name, xs[-1], ys[-1]))

    # Direct labels at the end of each line (text in ink, not the series colour)
    label_ys = spread_labels([y for _, _, y in end_points], min_gap=0.055)
    for (name, x, y), label_y in zip(end_points, label_ys):
        ax.annotate(
            f"{name}  {y:.0%}",
            xy=(x, label_y),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=TEXT_PRIMARY
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(runs) - 0.5 + max(0.6, len(runs) * 0.2))
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([run_label(run['run_id']) for run in runs])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)

    ax.set_title(
        "Label accuracy by prompt, per evaluation run",
        loc="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY, pad=28
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(prompt_names),
        frameon=False, fontsize=9, handlelength=3.5, markerscale=0.6, labelcolor=TEXT_SECONDARY, borderaxespad=0.2
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

def print_table(runs, metric: str = "label_accuracy"):
    """Same numbers as the chart, as a table."""
    print(f"\n{'Run':<18} {'Prompt':<16} {'Version':<10} {'Label Acc'}")
    print("-"*56)
    for run in runs:
        for summary in run['summaries']:
            name = summary['prompt_name']
            print(
                f"{run['run_id']:<18} "
                f"{name:<16} "
                f"{run['prompts'][name]['version']:<10} "
                f"{summary[metric]:.1%}"
            )

def main():
    runs = load_runs("runs")
    if not runs:
        print("No runs found in runs/. Run evaluation_system.py first.")
        return

    print_table(runs)

    output_file = os.path.join("charts", "accuracy_over_time.png")
    plot_accuracy_over_time(runs, output_file)
    print(f"\n📈 Chart saved to {output_file} ({len(runs)} run(s))")

if __name__ == "__main__":
    main()
