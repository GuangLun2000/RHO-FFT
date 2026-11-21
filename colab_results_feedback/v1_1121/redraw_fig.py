"""Redraw the accuracy and loss curves of the first P1 session with a legend that says what each curve is.

Reads the evaluation lines of the process log (results_1121.md) next to this file and writes fig_redrawn.png.
Usage: python redraw_fig.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"

# Encoding: hue = what the server does with the reports; line style = how clients are sampled;
# hollow markers without a line = the same arm with the zeroth-order updates switched off.
SOLID, DASHED = "-", (0, (5, 2.2))
ARMS = [
    ("参照方法", None),
    ("fedavg", dict(color=INK_2, ls=(0, (6, 1.5, 1.2, 1.5)), text="FedAvg：全部层做一阶训练（上限）")),
    ("decomfl", dict(color=GRAY, ls=(0, (1.2, 1.6)), text="DeComFL：全部层只做零阶更新（下限）")),
    ("HO-FL 原方法：一阶和零阶报告等权平均", None),
    ("A-b0", dict(color=BLUE, ls=SOLID, text="均匀采样（β=0）")),
    ("A-b0.75", dict(color=BLUE, ls=DASHED, text="偏向强设备采样（β=0.75，默认）")),
    ("B-b0", dict(color=BLUE, marker="o", text="均匀采样，关掉零阶更新")),
    ("B-b0.75", dict(color=BLUE, marker="s", text="偏向强设备采样，关掉零阶更新")),
    ("改聚合：每层只在一阶报告之间平均（fo_mean）", None),
    ("C-b0", dict(color=ORANGE, ls=SOLID, text="均匀采样（β=0）")),
    ("C-b0.75", dict(color=ORANGE, ls=DASHED, text="偏向强设备采样（β=0.75）")),
    ("D-b0", dict(color=ORANGE, marker="o", text="均匀采样，关掉零阶更新")),
    ("D-b0.75", dict(color=ORANGE, marker="s", text="偏向强设备采样，关掉零阶更新")),
    ("步长对照：原方法不变，只放大服务器步长（均匀采样）", None),
    ("Ax2-b0", dict(color=AQUA, ls=SOLID, marker="^", filled=True, text="步长 ×2")),
    ("Ax3-b0", dict(color=AQUA, ls=SOLID, marker="D", filled=True, text="步长 ×3")),
]


def use_cjk_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans GB", "PingFang SC", "Heiti TC", "STHeiti", "Songti SC", "Arial Unicode MS", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError("No font with Chinese glyphs found")


def read_curves(path):
    """{arm: {round: (accuracy %, test loss)}} from the 'eval @ round' lines of the 160-round Qwen runs."""
    pattern = re.compile(r"\[qwen-s42-([\w.\-]+)\] eval @ round (\d+): accuracy=([\d.]+) task_loss=([\d.]+)")
    curves = {}
    for match in pattern.finditer(path.read_text()):
        arm, round_id, accuracy, loss = match.group(1), int(match.group(2)), float(match.group(3)), float(match.group(4))
        curves.setdefault(arm, {})[round_id] = (100 * accuracy, loss)
    return curves


def style(spec):
    """Keyword arguments for ax.plot (also used for the legend sample)."""
    line = dict(color=spec["color"], linewidth=2.0, linestyle=spec.get("ls", "none"), solid_capstyle="round", zorder=3)
    if "marker" in spec:
        filled = spec.get("filled", False)
        line.update(marker=spec["marker"], markersize=6.5 if filled else 7, markeredgewidth=1.6,
                    markerfacecolor=spec["color"] if filled else SURFACE, markeredgecolor=SURFACE if filled else spec["color"],
                    zorder=4 if filled else 5)
        if filled:
            line.update(markeredgewidth=1.2)
    return line


def main():
    font = use_cjk_font()
    curves = read_curves(HERE / "results_1121.md")
    missing = [key for key, spec in ARMS if spec and key not in curves]
    if missing:
        raise RuntimeError(f"No evaluation lines for {missing}")

    figure = plt.figure(figsize=(17.5, 6.6), facecolor=SURFACE)
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 0.92], left=0.045, right=0.995, top=0.86, bottom=0.10, wspace=0.16)
    panels = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    for axis, index, title in zip(panels, (0, 1), ("测试准确率（%），越高越好", "测试损失，越低越好")):
        for key, spec in ARMS:
            if spec is None:
                continue
            rounds = sorted(curves[key])
            axis.plot(rounds, [curves[key][r][index] for r in rounds], **style(spec))
        axis.set_facecolor(SURFACE)
        axis.set_title(title, loc="left", fontsize=12.5, color=INK, pad=10)
        axis.set_xlabel("训练轮数", fontsize=11, color=INK_2)
        axis.set_xticks(range(0, 161, 20))
        axis.set_xlim(-3, 163)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_2, labelsize=10, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)

    # Direct labels only where a curve stands clear of the others; the end of training is shown enlarged.
    accuracy = panels[0]
    for x, y, text in [(22, 61.0, "原方法 + 均匀采样"), (22, 57.3, "DeComFL：160 轮没有进步")]:
        accuracy.text(x, y, text, fontsize=9.5, color=INK_2, ha="left", va="center", zorder=6,
                      bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2, alpha=0.85))
    zoom_x, zoom_y, box = (100, 162), (84.2, 89.0), (70, 59.6, 42, 14.2)  # box: x, y, width, height in data units
    inset = accuracy.inset_axes(box, transform=accuracy.transData, zorder=7)
    for key, spec in ARMS:
        if spec is not None and key != "decomfl":
            rounds = [r for r in sorted(curves[key]) if r >= zoom_x[0]]
            inset.plot(rounds, [curves[key][r][0] for r in rounds], **{**style(spec), "linewidth": 1.6, "markersize": 5})
    inset.set_xlim(*zoom_x)
    inset.set_ylim(*zoom_y)
    inset.set_xticks([100, 130, 160])
    inset.set_yticks([85, 86, 87, 88, 89])
    inset.set_facecolor(SURFACE)
    inset.grid(True, color=GRID, linewidth=0.6)
    inset.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in inset.spines.values():
        side.set_color("#c9c8c2")
    inset.set_title("最后 60 轮放大", fontsize=9.5, color=INK, loc="left", pad=4)
    accuracy.indicate_inset_zoom(inset, edgecolor="#a3a29c", linewidth=0.8, alpha=1.0)
    place = lambda value: box[1] + (value - zoom_y[0]) / (zoom_y[1] - zoom_y[0]) * box[3]  # zoomed y -> main-axis y
    final = lambda *keys: sum(curves[key][160][0] for key in keys) / len(keys)
    groups = [(final("Ax3-b0", "fedavg"), "步长 ×3，FedAvg"),
              (final("C-b0", "D-b0", "Ax2-b0"), "一阶投票（均匀采样），\n步长 ×2"),
              (final("A-b0", "A-b0.75", "B-b0", "B-b0.75", "C-b0.75", "D-b0.75"), "原方法 4 条，\n一阶投票（偏向采样）")]
    for value, text in groups:
        accuracy.text(box[0] + box[2] + 2.0, place(value), text, fontsize=9, color=INK_2, ha="left", va="center",
                      linespacing=1.25, zorder=8)

    # Legend drawn as a table: sample, plain description, arm code, final accuracy.
    legend = figure.add_subplot(grid[0, 2])
    legend.set_axis_off()
    legend.set_xlim(0, 1)
    rows = len(ARMS) + 2.6
    legend.set_ylim(rows, -0.6)
    legend.text(0.0, -0.25, "图例", fontsize=12.5, color=INK, va="center")
    legend.text(0.83, -0.25, "代号", fontsize=9.5, color=INK_2, va="center", ha="right")
    legend.text(1.0, -0.25, "第160轮\n准确率 %", fontsize=9.5, color=INK_2, va="center", ha="right", linespacing=1.15)
    y = 0.75
    for key, spec in ARMS:
        if spec is None:
            y += 0.28
            legend.text(0.0, y, key, fontsize=10.5, color=INK, va="center", fontweight="bold")
            legend.plot([0, 1], [y + 0.48, y + 0.48], color=GRID, linewidth=0.8)
        else:
            sample = style(spec)
            xs = [0.015, 0.065, 0.115] if sample["linestyle"] != "none" else [0.065]
            legend.plot(xs, [y] * len(xs), **{**sample, "markevery": [len(xs) // 2]})
            legend.text(0.15, y, spec["text"], fontsize=10, color=INK, va="center")
            legend.text(0.83, y, key, fontsize=9, color=INK_2, va="center", ha="right", family="monospace")
            legend.text(1.0, y, f"{curves[key][160][0]:.2f}", fontsize=10.5, color=INK, va="center", ha="right")
        y += 1.0
    legend.text(0.0, y + 0.25, "线型：实线 = 均匀采样，虚线 = 偏向强设备采样。\n空心标记（无连线）= 同一设置关掉零阶更新。\n"
                               "它们落在对应的线上：关掉零阶几乎不改变结果。", fontsize=9.5, color=INK_2, va="top", linespacing=1.5)

    figure.suptitle("Qwen2.5-1.5B / AG News（Dirichlet 0.5），种子 42，160 轮：12 种设置的测试曲线", x=0.045, y=0.965,
                    ha="left", fontsize=14.5, color=INK)
    out = HERE / "fig_redrawn.png"
    figure.savefig(out, dpi=170, facecolor=SURFACE)
    print("font:", font, "| written:", out)


if __name__ == "__main__":
    main()
