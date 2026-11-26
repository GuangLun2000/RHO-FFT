#!/usr/bin/env python
"""Figures of a session, drawn from ``feedback/feedback_E0.json`` alone.

    python experiments/sat/e0/plots.py <root>/feedback/feedback_E0.json [--out <directory>]

The Colab notebook loads this file by path and shows the figures in the page (Step 4). It needs matplotlib and the
standard library only, so it runs in the notebook's own Python and not in the environment of the pipeline; it
imports nothing of this repository. Every figure is saved as a PNG next to the report, and the numbers behind it
are in the tables of the feedback report.

What the figures keep fixed:

- A method has one colour and one marker in every figure (``METHOD_COLOURS``, ``METHOD_MARKERS``). The colours are
  the first slots of a categorical palette checked for colour-vision deficiency in this order; a method that is
  not drawn does not hand its colour to another one.
- One y-axis per chart. Two quantities of different scale get two charts side by side.
- A legend wherever two or more series are drawn, and the value at the end of every line, so that telling the
  series apart does not rest on colour alone. Text is never drawn in a series colour.
- The words inside a figure are English (the fonts of a Colab runtime have no Chinese glyphs); the heading the
  notebook prints above a figure is Chinese, like the rest of the page.
- In the figures of the method study a variant of the full method has a colour of its own, and its control (the
  same run with the zeroth-order part left out of the model) shares that colour as a dashed line with hollow
  markers. The colours drawn together in one figure were checked pair by pair for normal vision and for simulated
  colour-vision deficiency (OKLab distance; worst pair 10 under deficiency, 23 without, floors 8 and 15); the
  pairs that fail (blue with violet, orange with red, aqua with green) never share a figure.
"""
import json
import sys
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
# Categorical slots in their validated order: blue, orange, aqua, yellow, magenta, green, violet, red.
METHOD_COLOURS = {"m7": "#257de0", "m0": "#f47230", "m2": "#00b894", "m4": "#f2a600", "m1": "#ee7fac", "m3": "#008a17", "m5": "#6a4ce2", "m6": "#ec4647"}
METHOD_MARKERS = {"m7": "o", "m0": "s", "m2": "^", "m4": "D", "m1": "v", "m3": "P", "m5": "X", "m6": "*"}
METHOD_LABELS = {"m7": "Full method (RHO-FFT)", "m0": "Ideal centralised", "m2": "Backbone satellites only", "m4": "Zeroth-order flooding",
                 "m1": "Ground server", "m3": "Neighbour averaging", "m5": "Hybrid by node", "m6": "Static cuts"}
METHOD_SHORT = {"m7": "Full method", "m0": "Ideal", "m2": "Backbone only", "m4": "ZO flooding", "m1": "Ground", "m3": "Neighbours", "m5": "By node", "m6": "Static"}
TASK_LABELS = {"t1": "HumAID, Qwen2.5-1.5B", "t2": "EuroSAT, ViT-B/16"}
TASK_TITLES = {"t1": "HumAID", "t2": "EuroSAT"}
GROUP_TITLES = {"short": "短程运行", "day": "一整天的运行"}
FIRST_ORDER, ZEROTH_ORDER = "#257de0", "#f47230"
# The runs of the method study (``STUDY`` in e0.py): colour, marker, whether it is a control, legend label, label at the line's end.
STUDY_SERIES = {
    "t2-m2": ("#00b894", "^", False, "Backbone satellites only", "Backbone only"),
    "t2-m7": ("#257de0", "o", False, "Full method, rule of the brief", "Original rule"),
    "t2-m7-fo_mean": ("#ec4647", "v", False, "Full method, first-order mean over senders", "FO mean"),
    "t2-m7-fo_mean-zo_zeroed": ("#ec4647", "v", True, "the same, zeroth-order part left out", "FO mean, no ZO"),
    "t2-m7-line": ("#6a4ce2", "X", False, "Revised method (step-length search)", "Revised"),
    "t2-m7-line-zo_zeroed": ("#6a4ce2", "X", True, "Revised method, search not used", "No search"),
    "t2z-m2": ("#00b894", "^", False, "Backbone satellites only", "Backbone only"),
    "t2z-m7-line": ("#6a4ce2", "X", False, "Revised method (step-length search)", "Revised"),
    "t2z-m7-line-zo_zeroed": ("#6a4ce2", "X", True, "Revised method, search not used", "No search"),
    "t2z-m0": ("#f47230", "s", False, "Ideal centralised", "Ideal"),
}
STUDY_FIGURES = (("study_rule", ("t2-m2", "t2-m7", "t2-m7-fo_mean", "t2-m7-fo_mean-zo_zeroed"), "聚合规则：只改除数，以及不计入随机方向的零阶更新"),
                 ("study_search", ("t2-m2", "t2-m7-line", "t2-m7-line-zo_zeroed"), "步长搜索：修订后的方法、不用测出的步长、只让强星训练"))
STUDY_HELD = ("t2z-m2", "t2z-m7-line", "t2z-m7-line-zo_zeroed", "t2z-m0")
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
         "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
         "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
         "legend.fontsize": 9, "legend.frameon": True, "legend.framealpha": 1.0, "legend.fancybox": False, "legend.edgecolor": AXIS,
         "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "lines.linewidth": 1.8, "lines.markersize": 6,
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}
DPI = 200


def finished(run):
    return run.get("status") in ("COMPLETED", "DIVERGED") and bool(run.get("rounds"))


def new_figure(width, height, columns=1):
    figure = Figure(figsize=(width, height), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, columns, squeeze=False)[0]
    for ax in axes:
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
    return figure, list(axes)


def end_labels(ax, ends):
    """The value of every series at its right end, written to the right of the plot. ``ends`` is a list of
    ``(y, text)``. Labels that would overlap are moved apart and tied to their line by a thin connector."""
    low, high = ax.get_ylim()
    gap = 0.065 * (high - low)
    order = sorted(range(len(ends)), key=lambda index: ends[index][0])
    places = [ends[index][0] for index in order]
    for position in range(1, len(places)):                      # push apart from the bottom
        places[position] = max(places[position], places[position - 1] + gap)
    overflow = places[-1] - (high - 0.02 * (high - low)) if places else 0.0
    if overflow > 0:                                            # then back down, keeping the spacing
        places = [place - overflow for place in places]
        for position in range(len(places) - 2, -1, -1):
            places[position] = min(places[position], places[position + 1] - gap)
    across = ax.get_yaxis_transform()                           # x in axes units, y in data units
    for index, place in zip(order, places):
        value, text = ends[index]
        ax.plot([1.0, 1.025], [value, place], transform=across, color=AXIS, linewidth=0.8, clip_on=False, zorder=1)
        ax.text(1.035, place, text, transform=across, va="center", ha="left", fontsize=9, color=INK, clip_on=False)


def line_chart(title, ylabel, series, value_format, markers=True):
    """One chart of several methods over simulated time. ``series`` is a list of ``(method, hours, values)``."""
    figure, (ax,) = new_figure(7.0, 4.1)
    ends = []
    for method, hours, values in series:
        points = [(hour, value) for hour, value in zip(hours, values) if value is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=METHOD_COLOURS.get(method, MUTED), label=METHOD_LABELS.get(method, method), zorder=3,
                marker=METHOD_MARKERS.get(method, "o") if markers else None, markeredgecolor="white", markeredgewidth=1.0,
                markevery=max(1, len(xs) // 25))
        ends.append((ys[-1], f"{METHOD_SHORT.get(method, method)}  {value_format.format(ys[-1])}"))
    ax.set_xlabel("Simulated time (hours)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.02, y=0.08)
    ax.autoscale_view()
    ax.set_ylim(*ax.get_ylim())
    if len(ends) > 1:
        ax.legend(loc="best")
    end_labels(ax, ends)
    figure.subplots_adjust(left=0.10, right=0.74, top=0.92, bottom=0.13)
    return figure


def curve_figures(data):
    """Per task and group of runs: the validation accuracy at every evaluation and, for runs of a day, the training
    loss of every round (twelve rounds of four methods lie on top of each other and say nothing as a chart)."""
    result = []
    runs = [run for run in data.get("runs") or [] if finished(run)]
    for task in ("t2", "t1"):
        for group in ("short", "day"):
            chosen = [run for run in runs if run["task"] == task and run["group"] == group]
            chosen.sort(key=lambda run: list(METHOD_COLOURS).index(run["method"]) if run["method"] in METHOD_COLOURS else 99)
            scored = [run for run in chosen if run.get("curve")]
            if not scored:
                continue
            hours = scored[0]["hours"]
            where = f"{TASK_LABELS.get(task, task)}, {hours:g} simulated hours, seed 42, server step 1"
            figure = line_chart(where, "Validation accuracy (%)", [(run["method"], [point["sim_hours"] for point in run["curve"]],
                                                                   [100 * point["accuracy"] for point in run["curve"]]) for run in scored], "{:.1f}%")
            result.append((f"accuracy_{task}_{group}", f"{TASK_TITLES.get(task, task)} · {GROUP_TITLES[group]}：验证准确率随仿真时间的变化", figure))
            logged = [run for run in chosen if len(run.get("round_loss") or []) >= 24]
            if logged:
                per_hour = max(1.0, len(logged[0]["round_loss"]) / hours)
                figure = line_chart(where, "Training loss (mean over participants)",
                                    [(run["method"], [(index + 1) / per_hour for index in range(len(run["round_loss"]))], run["round_loss"]) for run in logged],
                                    "{:.3f}", markers=False)
                # Each method averages over its own participants, on their own data: the curves show that a run learns,
                # they do not rank the methods (validation accuracy does).
                result.append((f"loss_{task}_{group}", f"{TASK_TITLES.get(task, task)} · {GROUP_TITLES[group]}：每轮的训练损失"
                               "（各方法的参与者不同，曲线之间不能直接比较）", figure))
    return result


def styled_chart(title, ylabel, series, value_format):
    """One chart of runs told apart by a style of their own. ``series`` is a list of ``(style, hours, values)``
    with ``style`` an entry of ``STUDY_SERIES``: a control is dashed and has hollow markers."""
    figure, (ax,) = new_figure(7.4, 4.1)
    ends = []
    for (colour, marker, control, label, short), hours, values in series:
        points = [(hour, value) for hour, value in zip(hours, values) if value is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, label=label, zorder=3, linestyle="--" if control else "-", marker=marker,
                markerfacecolor="white" if control else colour, markeredgecolor=colour if control else "white", markeredgewidth=1.2 if control else 1.0,
                markevery=max(1, len(xs) // 25))
        ends.append((ys[-1], f"{short}  {value_format.format(ys[-1])}"))
    ax.set_xlabel("Simulated time (hours)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.02, y=0.08)
    ax.autoscale_view()
    ax.set_ylim(*ax.get_ylim())
    if len(ends) > 1:
        ax.legend(loc="best")
    end_labels(ax, ends)
    figure.subplots_adjust(left=0.095, right=0.76, top=0.92, bottom=0.13)
    return figure


def study_figures(data):
    """The method study: pairs of runs that differ in one thing, drawn together. Accuracy over simulated time for
    the change of the divisor and for the step-length search; with one class held by forward-only satellites
    alone, the accuracy over all classes and on that class; and what the search added round by round."""
    runs = {run["name"]: run for run in data.get("runs") or [] if run.get("group") == "study" and finished(run) and run.get("curve")}
    if not runs:
        return []
    result = []
    first = next(iter(runs.values()))
    where = f"{TASK_LABELS.get(first['task'], first['task'])}, {first['hours']:g} simulated hours, seed 42, server step 1"
    curve = lambda run, key: ([point["sim_hours"] for point in run["curve"]], [None if point.get(key) is None else 100 * point[key] for point in run["curve"]])  # noqa: E731
    for key, names, heading in STUDY_FIGURES:
        chosen = [runs[name] for name in names if name in runs]
        if len(chosen) >= 2:
            figure = styled_chart(where, "Validation accuracy (%)", [(STUDY_SERIES[run["name"]], *curve(run, "accuracy")) for run in chosen], "{:.1f}%")
            result.append((key, f"EuroSAT · {heading}", figure))
    held = [runs[name] for name in STUDY_HELD if name in runs]
    if len(held) >= 2:
        name = ", ".join(held[0].get("held_out_names") or []) or "one class"
        title = f"{TASK_LABELS.get(held[0]['task'], held[0]['task'])}, {name} held by forward-only satellites alone"
        figure = styled_chart(title, "Validation accuracy, all classes (%)", [(STUDY_SERIES[run["name"]], *curve(run, "accuracy")) for run in held], "{:.1f}%")
        result.append(("study_held_all", f"EuroSAT · {name} 只在推理芯片卫星上：全部类别的验证准确率", figure))
        if any(point.get("held_out_accuracy") is not None for run in held for point in run["curve"]):
            figure = styled_chart(title, f"Validation accuracy on {name} (%)", [(STUDY_SERIES[run["name"]], *curve(run, "held_out_accuracy")) for run in held], "{:.1f}%")
            result.append(("study_held_class", f"EuroSAT · {name} 只在推理芯片卫星上：这一类的验证准确率", figure))
    searched = runs.get("t2-m7-line")
    values = ((searched or {}).get("search") or {}).get("beta_by_round")
    if values:
        colour = STUDY_SERIES["t2-m7-line"][0]
        per_hour = max(1.0, len(values) / searched["hours"])
        window = max(1, round(per_hour))
        hours = [(index + 1) / per_hour for index in range(len(values))]
        smooth = [sum(values[max(0, index - window + 1):index + 1]) / len(values[max(0, index - window + 1):index + 1]) for index in range(len(values))]
        figure, (ax,) = new_figure(7.0, 3.6)
        ax.plot(hours, values, linestyle="none", marker="o", markersize=3.5, color=MUTED, alpha=0.55, label="each round", zorder=2)
        ax.plot(hours, smooth, color=colour, label=f"mean of the last {window} rounds", zorder=3)
        ax.set_xlabel("Simulated time (hours)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_ylabel("Previous update added (multiples of it)")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title("Revised method: what the step-length search of the forward-only satellites added")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)   # below the plot: the points fill all of it
        figure.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.27)
        result.append(("study_search_steps", "EuroSAT · 修订后的方法：步长搜索每轮加上了多少个上一轮的更新（0 到 1）", figure))
    return result


def memory_figure(data):
    """Peak GPU memory of one participation at every cut, one panel per model: the cut table of the scenario."""
    measured = [(task, memory) for task, memory in sorted((data.get("memory") or {}).items()) if memory and memory.get("rows")]
    measured = [(task, memory) for task, memory in measured if any(row["mode"] == "hybrid" and row["peak_allocated_gb"] is not None for row in memory["rows"])]
    if not measured:
        return []
    figure, axes = new_figure(4.9 * len(measured), 4.0, len(measured))
    for ax, (task, memory) in zip(axes, measured):
        rows = sorted((row for row in memory["rows"] if row["mode"] == "hybrid" and row["peak_allocated_gb"] is not None), key=lambda row: row["cut"])
        cuts, peaks = [row["cut"] for row in rows], [row["peak_allocated_gb"] for row in rows]
        roles = {0: "backbone", len(cuts) - 1: "forward-\nonly"}
        if len(cuts) >= 4:
            roles.update({1: "hybrid\n(idle)", 2: "hybrid\n(imaging)"})
        positions = list(range(len(cuts)))
        ax.bar(positions, peaks, width=0.5, color=FIRST_ORDER, zorder=3)
        for position, peak in zip(positions, peaks):
            ax.text(position, peak, f"{peak:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        inference = next((row["peak_allocated_gb"] for row in memory["rows"] if row["mode"] == "inference" and row["peak_allocated_gb"] is not None), None)
        if inference is not None:   # a reference level, not a series: dashed, and named in the free corner of the plot
            ax.axhline(inference, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
            ax.text(0.98, 0.95, f"dashed: one forward pass without training, {inference:.2f} GB", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=MUTED)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{cut}\n{roles.get(index, '')}" for index, cut in enumerate(cuts)], fontsize=9)
        ax.set_xlabel("Cut: the blocks below it are forward-only")
        ax.set_ylabel("Peak GPU memory (GB)")
        ax.set_title(f"{TASK_LABELS.get(task, task)}, batch {'×'.join(str(size) for size in memory.get('batch_shape') or [])}")
        ax.set_ylim(0, max(peaks) * 1.22)
    figure.subplots_adjust(left=0.09 if len(measured) > 1 else 0.14, right=0.98, top=0.91, bottom=0.26, wspace=0.28)
    return [("memory", "一次参与的峰值显存（按分界）", figure)]


def channel_figures(data):
    """The full method, block by block: who sends which kind of update, and how large the two parts of the update
    are. Drawn from the longest finished run of the full method on each task that logged the statistics, under the
    rule of the brief and on the usual data: a variant changes who sends what, and has its own figures."""
    result = []
    for task in ("t2", "t1"):
        runs = [run for run in data.get("runs") or [] if finished(run) and run["task"] == task and run["method"] == "m7" and run.get("channels")
                and not run.get("variant")]
        if not runs:
            continue
        run = max(runs, key=lambda entry: entry["rounds"])
        stats, depth = run["channels"], run.get("depth")
        blocks = stats["blocks"]
        names = ["head" if block == depth else str(block) for block in blocks]
        positions = list(range(len(blocks)))
        figure, (left, right) = new_figure(11.0, 4.4, 2)
        first, zeroth = [100 * value for value in stats["fo_share"]], [100 * value for value in stats["zo_share"]]
        left.bar(positions, first, width=0.62, color=FIRST_ORDER, edgecolor="white", linewidth=1.0, label="first-order update (backpropagation)", zorder=3)
        left.bar(positions, zeroth, width=0.62, bottom=first, color=ZEROTH_ORDER, edgecolor="white", linewidth=1.0, label="zeroth-order update (forward-only)", zorder=3)
        left.set_ylim(0, 100)
        left.set_ylabel("Share of the participants of a round (%)")
        left.set_title("Who sends which update to a block")
        norms = [("first-order part", stats["fo_norm"], FIRST_ORDER, "o"), ("zeroth-order part", stats["zo_norm"], ZEROTH_ORDER, "s")]
        for label, values, colour, marker in norms:
            points = [(position, value) for position, value in zip(positions, values) if value and value > 0]
            if points:
                right.plot(*zip(*points), color=colour, marker=marker, markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)
        right.set_yscale("log")
        right.grid(True, axis="y", which="both")
        right.set_ylabel("Norm of this part of the mean update")
        right.set_title("How large the two parts of the update are")
        step = max(1, len(blocks) // 14)
        for ax in (left, right):
            ax.set_xticks(positions[::step])
            ax.set_xticklabels(names[::step])
            ax.set_xlabel("Transformer block (0 is the lowest)")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)   # below the plot: inside it would cover marks
        figure.suptitle(f"{TASK_LABELS.get(task, task)}, full method, mean over {stats['rounds']} rounds", fontsize=11, color=INK)
        figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.27, wspace=0.22)
        result.append((f"channels_{task}", f"{TASK_TITLES.get(task, task)} · 完整方法：两条通道对每一层更新的贡献", figure))
    return result


def figures(data, out_dir):
    """Draw every figure the data allows and save it under ``out_dir``. Returns ``[(heading, path, error)]`` in the
    order they should be shown: ``path`` is the PNG, or ``None`` with the error when a figure could not be drawn
    (one figure that fails does not take the others with it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = []
    with matplotlib.rc_context(STYLE):
        for build in (curve_figures, study_figures, channel_figures, memory_figure):
            try:
                drawn = build(data)
            except Exception as error:  # noqa: BLE001
                shown.append((build.__name__, None, repr(error)))
                continue
            for key, heading, figure in drawn:
                path = out_dir / f"{key}.png"
                try:
                    figure.savefig(path, dpi=DPI)
                    shown.append((heading, path, None))
                except Exception as error:  # noqa: BLE001
                    shown.append((heading, None, repr(error)))
    return shown


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    source = Path(argv[0])
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else source.parent / "figures"
    for heading, path, error in figures(json.loads(source.read_text(encoding="utf-8")), out_dir):
        print(f"{heading}: {path if error is None else 'not drawn, ' + error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
