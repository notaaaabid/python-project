import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend; Tkinter draws it
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE     = ["#4C9BE8", "#E85C4C", "#4CE89B"]  # blue / red / green
DARK_BG     = "#1A1F2E"
CARD_BG     = "#242938"
TEXT_COLOR  = "#E8ECF4"
ACCENT      = "#4C9BE8"

def _apply_dark_theme(fig: plt.Figure, axes):
    """Apply consistent dark theme to any figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3A4055")


# ── 1. Disorder Distribution Pie ─────────────────────────────────────────────
def plot_disorder_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df["Sleep Disorder"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    wedge_props = dict(width=0.55, edgecolor=DARK_BG, linewidth=2)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(counts)],
        wedgeprops=wedge_props,
        startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(DARK_BG)
        at.set_fontweight("bold")
    ax.set_title("Sleep Disorder Distribution", color=TEXT_COLOR, fontsize=12, pad=12)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 2. Sleep Duration Histogram ───────────────────────────────────────────────
def plot_sleep_duration(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for disorder, color in zip(df["Sleep Disorder"].unique(), PALETTE):
        subset = df[df["Sleep Disorder"] == disorder]["Sleep Duration"]
        ax.hist(subset, bins=14, alpha=0.70, color=color,
                label=disorder, edgecolor=DARK_BG, linewidth=0.5)
    ax.set_xlabel("Sleep Duration (hours)")
    ax.set_ylabel("Count")
    ax.set_title("Sleep Duration by Disorder", fontsize=12)
    ax.legend(facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT_COLOR, fontsize=9)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric = df.select_dtypes(include=[np.number])
    corr    = numeric.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", linewidths=0.4,
        cmap="coolwarm", center=0,
        annot_kws={"size": 7, "color": TEXT_COLOR},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=12)
    for text in ax.texts:
        text.set_color(TEXT_COLOR)
        text.set_fontsize(11)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 5. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(fi: pd.Series) -> plt.Figure:
    top = fi.head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [ACCENT if i == 0 else "#3A6EA5" for i in range(len(top))]
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=colors[::-1], edgecolor=DARK_BG, linewidth=0.5)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importances", fontsize=12)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color=TEXT_COLOR, fontsize=8)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 6. Model Comparison Bar Chart ─────────────────────────────────────────────
def plot_model_comparison(all_scores: dict) -> plt.Figure:
    names  = list(all_scores.keys())
    scores = [all_scores[n] * 100 for n in names]
    colors = [ACCENT if s == max(scores) else "#3A6EA5" for s in scores]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, scores, color=colors, edgecolor=DARK_BG, linewidth=0.5,
                  width=0.55)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{s:.1f}%", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9, fontweight="bold")
    ax.set_ylabel("CV Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison (5-Fold CV)", fontsize=12)
    ax.tick_params(axis="x", rotation=12)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 7. Stress vs Sleep Quality Scatter ───────────────────────────────────────
def plot_stress_vs_quality(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for disorder, color in zip(df["Sleep Disorder"].unique(), PALETTE):
        sub = df[df["Sleep Disorder"] == disorder]
        ax.scatter(sub["Stress Level"], sub["Quality of Sleep"],
                   alpha=0.55, color=color, label=disorder, s=22, edgecolors="none")
    ax.set_xlabel("Stress Level (1–9)")
    ax.set_ylabel("Quality of Sleep (1–9)")
    ax.set_title("Stress Level vs Sleep Quality", fontsize=12)
    ax.legend(facecolor=CARD_BG, edgecolor=ACCENT, labelcolor=TEXT_COLOR, fontsize=9)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── 8. Probability Gauge Bar (single prediction) ─────────────────────────────
def plot_probability_gauge(prob_dict: dict) -> plt.Figure:
    """
    Horizontal bar chart showing predicted class probabilities for a
    single inference call.
    """
    labels  = list(prob_dict.keys())
    values  = [prob_dict[l] * 100 for l in labels]
    colors_map = {"None": "#4CE89B", "Sleep Apnea": "#E85C4C", "Insomnia": "#F0A500"}
    bar_colors = [colors_map.get(l, ACCENT) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.barh(labels, values, color=bar_colors,
                   edgecolor=DARK_BG, linewidth=0.5, height=0.45)
    for bar, v in zip(bars, values):
        ax.text(min(v + 1, 95), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", color=TEXT_COLOR, fontsize=10,
                fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Probabilities", fontsize=11)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    fig = plot_disorder_distribution(df)
    fig.savefig("/tmp/test_chart.png", dpi=100)
    print("Chart saved to /tmp/test_chart.png")