#!/usr/bin/env python
"""Dose-response figure of the P2 feedback: for each fault strength, the influence of the unreliable report on the
scarce segment and the damage to the test loss, against the number of honest first-order reports next to it.

    python experiments/p2/plot_dose.py <feedback_P2.json> <output directory>

Needs matplotlib only, so it runs in the notebook's own Python.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Aggregation rules in a fixed order with fixed colours; dashed = runs without zeroth-order updates.
RULES = [("mean_all", "#2a78d6", "-", "o", "equal-weight mean of all reports (HO-FL)"),
         ("fo_diluted", "#8a8983", (0, (1.2, 1.6)), "o", "first-order reports only, divided by K"),
         ("fo_mean", "#eb6834", "-", "s", "mean of the first-order reports only"),
         ("robust_all", "#1baf7a", "-", "^", "geometric median of all reports"),
         ("fo_robust", "#eda100", "-", "D", "geometric median of the first-order reports"),
         ("quorum_robust", "#e87ba4", "-", "P", "quorum switching (first-order if >= 3, else all)"),
         ("quorum_hold", "#008300", (0, (5, 2.2)), "v", "quorum switching, no zeroth-order: hold below the quorum"),
         ("quorum_dilute", "#6250d6", (0, (5, 2.2)), "X", "quorum switching, no zeroth-order: dilute below the quorum")]


def style_axis(axis, title, doses):
    axis.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
    axis.set_xlabel("m = honest first-order reports on S1 next to the unreliable one (quorum of 3 reached from m = 2)", fontsize=9.5, color=INK_2)
    axis.set_xticks(doses)
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_2, labelsize=9.5, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.axvline(1.5, color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)))


def cf_figure(data, out_dir):
    """Counterfactual design: progress and influence on S1 against the dose, one row per fault strength."""
    cells = data["cf"]["summary"]["table"]
    doses = sorted({cell["m"] for cell in cells})
    faults = ["none", "flip1", "flip20"]
    height = 4.4 * len(faults) + 1.3
    figure, axes = plt.subplots(len(faults), 2, figsize=(15, height), squeeze=False, facecolor=SURFACE)
    for row, fault in enumerate(faults):
        for column, (key, title) in enumerate((("progress", "progress on S1 (1 = one honest first-order report)"),
                                               ("influence", "relative influence of the faulty report (1 = equal-weight mean)"))):
            axis = axes[row][column]
            if fault == "none" and key == "influence":
                axis.set_axis_off()
                axis.text(0.02, 0.5, "No fault: influence is not defined.\nLeft: how much each rule lowers the evaluation loss on S1\nwhen every report is honest.",
                          fontsize=10, color=INK_2, va="center")
                continue
            for name, color, line, marker, text in RULES:
                points = sorted((cell["m"], cell[key]) for cell in cells if cell["fault"] == fault and cell["rule"] == name and cell[key] is not None)
                if points:
                    axis.plot([m for m, _ in points], [value for _, value in points], color=color, linestyle=line, marker=marker, markersize=7,
                              markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2.0, label=text)
            if key == "influence":
                axis.set_yscale("symlog", linthresh=0.1)
                axis.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
            else:
                axis.axhline(0.0, color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
            style_axis(axis, f"{fault}: {title}", doses)
    handles, labels = axes[1][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False, labelcolor=INK)
    figure.suptitle(f"{data['model']} / AG News (Dirichlet 0.5), seed {data['seed']}: counterfactual dose-response "
                    f"({data['cf']['summary']['rounds']} rounds, every cell from the same reports)", x=0.02, ha="left", fontsize=13, color=INK)
    figure.tight_layout(rect=(0, 1.35 / height, 1, 1 - 0.35 / height))
    out = Path(out_dir) / "p2_dose_cf.png"
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    print("written:", out)
    return out


def main(feedback_json, out_dir):
    data = json.loads(Path(feedback_json).read_text())
    if data.get("cf"):
        cf_figure(data, out_dir)
    tables = data["tables"]
    if not any(cell for rows in tables.values() for row in rows for cell in row["cells"]):
        print("no full-arm run finished")
        return None
    height = 4.9 * len(tables) + 1.3   # the legend below the panels needs a fixed strip
    figure, axes = plt.subplots(len(tables), 2, figsize=(15, height), squeeze=False, facecolor=SURFACE)
    for row, (fault, rows) in enumerate(tables.items()):
        by_rule = {entry["aggregation"]: [cell for cell in entry["cells"] if cell] for entry in rows}
        for column, (key, title) in enumerate((("influence_s1", "relative influence on S1 (1 = equal-weight mean)"),
                                               ("damage", "damage: test loss with the fault minus paired clean run"))):
            axis = axes[row][column]
            for name, color, line, marker, text in RULES:
                cells = [cell for cell in by_rule.get(name, []) if cell.get(key) is not None]
                if cells:
                    axis.plot([cell["m"] for cell in cells], [cell[key] for cell in cells], color=color, linestyle=line, marker=marker,
                              markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2.0, label=text)
            if column == 0:
                axis.set_yscale("symlog", linthresh=0.1)
            axis.axhline(1.0 if column == 0 else 0.0, color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
            axis.axvline(1.5, color=INK_2, linewidth=0.8, linestyle=(0, (1, 2)))
            axis.set_title(f"{fault}: {title}", loc="left", fontsize=11.5, color=INK, pad=8)
            axis.set_xlabel("m = honest first-order reports on S1 next to the unreliable one (quorum of 3 reached from m = 2)", fontsize=9.5, color=INK_2)
            axis.set_xticks(data["doses"])
            axis.set_facecolor(SURFACE)
            axis.grid(True, color=GRID, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.tick_params(colors=INK_2, labelsize=9.5, length=0)
            for side in ("top", "right", "left"):
                axis.spines[side].set_visible(False)
            axis.spines["bottom"].set_color(GRID)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False, labelcolor=INK)
    figure.suptitle(f"{data['model']} / AG News (Dirichlet 0.5), seed {data['seed']}: dose-response of one unreliable first-order source "
                    f"({data['dose_rounds']} rounds per arm)", x=0.02, ha="left", fontsize=13, color=INK)
    figure.tight_layout(rect=(0, 1.35 / height, 1, 1 - 0.35 / height))
    out = Path(out_dir) / "p2_dose.png"
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    print("written:", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
