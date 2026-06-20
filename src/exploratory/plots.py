import re
from pathlib import Path
import textwrap

import polars as pl
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns

from src.exploratory.explore_data import resume_modalites, resume_target_dpe


def plot_pareto_modalites(
    df: pl.DataFrame,
    col: str,
    top_n: int | None = None,
    verbose: bool = True,
    output_dir: str = "src/plots/cat",
    seuil: float = 80,
):
    mpl.rcdefaults()

    data = resume_modalites(df, col)

    if top_n is not None:
        data = data.head(top_n)

    data = data.to_pandas()

    labels = data[col].astype(str)
    labels = [
        "\n".join(textwrap.wrap(label, width=22))
        for label in labels
    ]

    pourcentages = data["pourcentage"]
    pourcentages_cumules = data["pourcentage_cumule"]

    safe_col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)
    filename = f"pareto_{safe_col}.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / filename

    with plt.style.context("default"):
        fig, ax1 = plt.subplots(figsize=(12, 6.5), facecolor="white")

        fig.patch.set_facecolor("white")
        ax1.set_facecolor("#F8FAFC")

        bars = ax1.bar(
            labels,
            pourcentages,
            color="#5B8DB8",
            edgecolor="#2F5D7C",
            linewidth=0.8,
            width=0.65,
            alpha=0.95,
            label="Part des modalités",
        )

        ax2 = ax1.twinx()
        ax2.set_facecolor("none")

        ax2.plot(
            labels,
            pourcentages_cumules,
            color="#D62828",
            marker="o",
            linewidth=2.5,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor="#D62828",
            label="Pourcentage cumulé",
        )

        ax2.axhline(
            seuil,
            color="#F77F00",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )

        ax2.text(
            len(labels) - 0.55,
            seuil + 1.5,
            f"Seuil {seuil:.0f} %",
            color="#F77F00",
            fontsize=10,
            ha="right",
            va="bottom",
        )

        # IMPORTANT : mêmes graduations et même échelle sur les deux axes
        y_ticks = [0, 20, 40, 60, 80, 100]

        ax1.set_ylim(0, 105)
        ax2.set_ylim(0, 105)

        ax1.set_yticks(y_ticks)
        ax2.set_yticks(y_ticks)

        ax1.set_ylabel("Part des modalités (%)", fontsize=11, color="#222222")
        ax2.set_ylabel("Pourcentage cumulé (%)", fontsize=11, color="#222222")

        ax1.set_xlabel(col, fontsize=11, color="#222222", labelpad=12)

        ax1.tick_params(axis="x", rotation=0, colors="#222222", labelsize=10)
        ax1.tick_params(axis="y", colors="#222222", labelsize=10)
        ax2.tick_params(axis="y", colors="#222222", labelsize=10)

        ax1.grid(
            axis="y",
            color="#E1E5EA",
            linestyle="-",
            linewidth=0.9,
            alpha=0.9,
        )
        ax1.grid(axis="x", visible=False)

        for spine in ["top"]:
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)

        ax1.spines["left"].set_color("#D0D5DD")
        ax1.spines["bottom"].set_color("#D0D5DD")
        ax2.spines["right"].set_color("#D0D5DD")

        for bar, pct in zip(bars, pourcentages):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f} %",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#1F2937",
                fontweight="bold",
            )

        titre_col = col.replace("_", " ")

        ax1.set_title(
            f"Diagramme de Pareto — {titre_col}",
            fontsize=16,
            fontweight="bold",
            color="#1F2937",
            pad=18,
        )

        fig.tight_layout()

        fig.savefig(
            fig_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )

        if verbose:
            plt.show()
        else:
            plt.close(fig)

    return fig_path


## variables numériques

def plot_boxplot_and_hist(
    df: pl.DataFrame,
    col: str,
    output_dir: str = "src/plots/num",
    verbose: bool = True,
):
    """
    Affiche et sauvegarde le boxplot et la distribution d'une variable quantitative.
    """

    values = df.get_column(col).drop_nulls().to_numpy()

    safe_col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)
    filename = f"boxplot_distribution_{safe_col}.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / filename

    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [0.25, 0.75]},
        facecolor="white",
    )

    fig.patch.set_facecolor("white")
    ax_box.set_facecolor("#F8FAFC")
    ax_hist.set_facecolor("#F8FAFC")

    sns.boxplot(
        x=values,
        ax=ax_box,
        color="#5B8DB8",
        linewidth=1.2,
    )

    sns.histplot(
        x=values,
        ax=ax_hist,
        bins=40,
        kde=True,
        color="#5B8DB8",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )

    ax_box.set_xlabel("")
    ax_box.set_ylabel("")

    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Effectif")

    ax_hist.grid(
        axis="y",
        color="#E1E5EA",
        linestyle="-",
        linewidth=0.8,
        alpha=0.9,
    )

    ax_hist.grid(axis="x", visible=False)

    for ax in [ax_box, ax_hist]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D0D5DD")
        ax.spines["bottom"].set_color("#D0D5DD")
        ax.tick_params(colors="#222222")

    fig.suptitle(
        f"Boxplot et distribution — {col}",
        fontsize=16,
        fontweight="bold",
        color="#1F2937",
        y=0.98,
    )

    fig.tight_layout()

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    if verbose:
        plt.show()
    else:
        plt.close(fig)

    return fig_path

## TARGET

def plot_target_dpe(
    df: pl.DataFrame,
    classes_dpe: list,
    dpe_colors: dict,
    target_col: str = "etiquette_dpe",
    output_dir: str = "src/plots/cat",
    verbose: bool = True,
):
    resume = resume_target_dpe(df, classes_dpe, target_col)

    data = resume.to_pandas()

    labels = data[target_col].astype(str)
    effectifs = data["effectif"]
    pourcentages = data["pourcentage"]

    colors = [dpe_colors[label] for label in labels]

    safe_col = re.sub(r"[^a-zA-Z0-9_]+", "_", target_col)
    filename = f"distribution_{safe_col}.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / filename

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    bars = ax.bar(
        labels,
        pourcentages,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        width=0.65,
    )

    ax.set_title(
        "Distribution de la classe DPE",
        fontsize=16,
        fontweight="bold",
        color="#1F2937",
        pad=18,
    )

    ax.set_xlabel("Classe DPE", fontsize=11, color="#222222")
    ax.set_ylabel("Pourcentage (%)", fontsize=11, color="#222222")

    ax.set_ylim(0, max(pourcentages) * 1.20)

    ax.grid(
        axis="y",
        color="#E1E5EA",
        linestyle="-",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#D0D5DD")
    ax.spines["bottom"].set_color("#D0D5DD")

    ax.tick_params(axis="x", colors="#222222", labelsize=11)
    ax.tick_params(axis="y", colors="#222222", labelsize=10)

    for bar, pct, eff in zip(bars, pourcentages, effectifs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pourcentages) * 0.02,
            f"{pct:.1f} %\n({eff})",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
            fontweight="bold",
        )

    fig.tight_layout()

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    if verbose:
        plt.show()
    else:
        plt.close(fig)

    return fig_path


def plot_dist(
    df: pl.DataFrame,
    col: str,
    positive_only: bool = False,
    output_dir: str = "src/plots/num",
    verbose: bool = True,
):
    """
    Cette fonction trace et sauvegarde l'histogramme d'une variable numérique.

    Paramètres :
    - positive_only=True : conserve uniquement les valeurs strictement positives.
    - output_dir : dossier de sauvegarde du graphique.
    - verbose=True : affiche le graphique.
    """

    values = df.get_column(col).drop_nulls()

    if positive_only:
        values = values.filter(values > 0)

    values = values.to_numpy()

    if len(values) == 0:
        raise ValueError(f"Aucune valeur disponible pour tracer la distribution de {col}.")

    safe_col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)

    suffix = "_positives" if positive_only else ""
    filename = f"distribution_{safe_col}{suffix}.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / filename

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    sns.histplot(
        values,
        bins=50,
        kde=False,
        ax=ax
    )

    titre = f"Distribution de {col}"
    if positive_only:
        titre += " — valeurs positives uniquement"

    ax.set_title(titre)
    ax.set_xlabel(col)
    ax.set_ylabel("Effectif")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    if verbose:
        plt.show()
    else:
        plt.close(fig)

    return fig_path


def boxplot_vs_target(
    df: pl.DataFrame,
    col: str,
    target: str = "",
    target_order: list = [],
    output_dir: str = "src/plots/bivariee",
    verbose: bool = True,
):
    """
    Trace un boxplot d'une variable numérique selon la target.
    """

    data = (
        df.select([col, target])
        .drop_nulls()
        .to_pandas()
    )

    safe_col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)
    filename = f"boxplot_{safe_col}_vs_{target}.png"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig_path = output_path / filename

    plt.figure(figsize=(10, 6), facecolor="white")

    sns.boxplot(
        data=data,
        x=target,
        y=col,
        order=target_order,
        showfliers=True
    )

    plt.title(f"{col} selon la classe DPE")
    plt.xlabel("Classe DPE regroupée")
    plt.ylabel(col)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    if verbose:
        plt.show()
    else:
        plt.close()

    return fig_path


def plot_missing_train_test(
    X_train,
    X_test,
    only_missing: bool = True,
    top_n: int | None = None,
    output_dir: str = "src/plots/missing",
    verbose: bool = True,
):
    """
    Compare le taux de valeurs manquantes entre train et test.
    Fonction compatible avec Polars ou Pandas.
    """

    # Conversion Polars -> Pandas si nécessaire
    X_train_pd = X_train.to_pandas() if hasattr(X_train, "to_pandas") else X_train
    X_test_pd = X_test.to_pandas() if hasattr(X_test, "to_pandas") else X_test

    missing_train = X_train_pd.isna().mean() * 100
    missing_test = X_test_pd.isna().mean() * 100

    missing_df = pd.DataFrame(
        {
            "Train": missing_train,
            "Test": missing_test,
        }
    )

    missing_df["max_missing"] = missing_df[["Train", "Test"]].max(axis=1)

    if only_missing:
        missing_df = missing_df[missing_df["max_missing"] > 0]

    missing_df = missing_df.sort_values("max_missing", ascending=True)

    if top_n is not None:
        missing_df = missing_df.tail(top_n)

    missing_df = missing_df.drop(columns="max_missing")

    if missing_df.empty:
        raise ValueError("Aucune valeur manquante à afficher.")

    fig_height = max(6, 0.35 * len(missing_df))

    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")

    missing_df.plot(
        kind="barh",
        ax=ax,
        width=0.75,
        edgecolor="#333333",
        linewidth=0.6,
    )

    ax.set_title(
        "Taux de valeurs manquantes — Train vs Test",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Valeurs manquantes (%)")
    ax.set_ylabel("Variables")

    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="Jeu de données")

    max_value = missing_df.max().max()
    ax.set_xlim(0, max_value * 1.15 if max_value > 0 else 1)

    # Afficher les pourcentages au bout des barres
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
            fontsize=8,
        )

    fig.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = "missing_values_train_test.png"
    fig_path = output_path / filename

    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    if verbose:
        plt.show()
    else:
        plt.close(fig)

    return fig_path