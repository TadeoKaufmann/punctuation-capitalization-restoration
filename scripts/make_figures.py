"""Extra figures and the majority-class reference, built only from the CSVs in results/final/.

    python scripts/make_figures.py

Writes figures/f1_by_source.png, figures/f1_by_length.png and results/final/majority_baseline.csv.
The majority baseline is not a trained model: it is the macro F1 of always predicting the most frequent class,
derived from the class counts of the test set.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results" / "final", ROOT / "figures"

MODELS = ["RF sin embeddings", "RF con embeddings", "LSTM", "BiLSTM"]
LABELS = {"RF sin embeddings": "RF, handcrafted features", "RF con embeddings": "RF, embeddings",
          "LSTM": "LSTM", "BiLSTM": "BiLSTM"}
COLORS = dict(zip(MODELS, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
TASKS = {"Puntuación inicial": "Opening mark (¿)", "Puntuación final": "Closing mark (, . ?)",
         "Capitalización": "Capitalization"}
SOURCES = {"conferencias": "Press briefings", "subtitulos": "Film subtitles", "wikinoticias": "Wikinews",
           "paenza": "Popular maths books", "sintetico": "Synthetic speech"}
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def load_breakdown(name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / name)
    df.columns = ["group", "task"] + list(df.columns[2:])
    return df


def style(ax):
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def figure_by_source():
    df = load_breakdown("f1_por_fuente.csv")
    df["key"] = df["group"].str.split(" ").str[0]
    order = list(SOURCES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    height = 0.2
    for ax, (task, title) in zip(axes, TASKS.items()):
        part = df[df["task"] == task].set_index("key").loc[order]
        y = np.arange(len(order))
        for i, model in enumerate(MODELS):
            ax.barh(y + (i - 1.5) * height, part[model], height=height * 0.86, color=COLORS[model], label=LABELS[model])
        for row, value in zip(y, part["BiLSTM"]):
            ax.text(value + 0.012, row + 1.5 * height, f"{value:.2f}", va="center", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
        ax.set_xlim(0, 1)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        style(ax)
    axes[0].set_yticks(np.arange(len(order)), [SOURCES[s] for s in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Test macro F1 by text source (BiLSTM values labelled)", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.text(0.01, 0.075, "The Wikinews test text has a single ¿ and every model misses it, hence the flat 0.50 in that panel.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(FIGURES / "f1_by_source.png", dpi=150)
    plt.close(fig)


def figure_by_length():
    df = load_breakdown("f1_por_largo.csv")
    df["key"] = df["group"].str.split(" ").str[0].str.replace("-+", "+", regex=False)
    order = ["1-5", "6-10", "11-20", "21-40", "41+"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, (task, title) in zip(axes, TASKS.items()):
        part = df[df["task"] == task].set_index("key").loc[order]
        for model in MODELS:
            ax.plot(order, part[model], color=COLORS[model], linewidth=2, marker="o", markersize=6, label=LABELS[model])
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("words per instance", fontsize=9, color=MUTED)
        ax.set_ylim(0.3, 0.9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        style(ax)
    axes[0].set_ylabel("test macro F1", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Test macro F1 by instance length", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(FIGURES / "f1_by_length.png", dpi=150)
    plt.close(fig)


def majority_baseline() -> pd.DataFrame:
    per_class = pd.read_csv(RESULTS / "f1_por_clase.csv")
    counts = per_class[per_class["modelo"] == "BiLSTM"]
    rows = []
    for task, part in counts.groupby("tarea", sort=False):
        share = part["palabras"].max() / part["palabras"].sum()
        # always predicting the top class: precision = share, recall = 1; every other class scores 0
        rows.append({"task": TASKS[task], "classes": len(part), "majority_share": round(share, 4),
                     "macro_f1": round(2 * share / (1 + share) / len(part), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "majority_baseline.csv", index=False, lineterminator="\n")
    return out


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    figure_by_source()
    figure_by_length()
    table = majority_baseline()
    print(table.to_string(index=False))
    print("mean macro F1:", round(table["macro_f1"].mean(), 4))
