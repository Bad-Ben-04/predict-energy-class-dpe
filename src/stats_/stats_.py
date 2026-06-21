import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import chi2_contingency, kruskal


def test_chi2_cramers_v(
        df: pl.DataFrame,
        col: str,
        target: str,
        valeur_missing: str = "Non renseigné",
) -> dict:
    """
    Test d'association entre une variable qualitative et la target.

    - Les NaN sont remplacés temporairement par "Non renseigné".
    - Retourne la p-value et le V de Cramer.
    """

    data = (
        df.select([col, target])
        .with_columns([
            pl.col(col).cast(pl.Utf8).fill_null(valeur_missing),
            pl.col(target).cast(pl.Utf8).fill_null(valeur_missing),
        ])
        .to_pandas()
    )

    table = pd.crosstab(data[col], data[target])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "variable": col,
            "type_variable": "qualitative",
            "test": "chi2",
            "n": len(data),
            "nb_modalites": table.shape[0],
            "statistique": np.nan,
            "p_value": np.nan,
            "taille_effet": np.nan,
            "effet": "cramers_v",
            "interpretation": "non_testable",
        }

    chi2, p_value, dof, expected = chi2_contingency(table)

    n = table.to_numpy().sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)

    cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan

    if cramers_v < 0.10:
        interpretation = "faible"
    elif cramers_v < 0.30:
        interpretation = "moderee"
    else:
        interpretation = "forte"

    return {
        "variable": col,
        "type_variable": "qualitative",
        "test": "chi2",
        "n": int(n),
        "nb_modalites": table.shape[0],
        "statistique": round(chi2, 4),
        "p_value": p_value,
        "taille_effet": round(cramers_v, 4),
        "effet": "cramers_v",
        "interpretation": interpretation,
    }


def test_kruskal_epsilon(
        df: pl.DataFrame,
        col: str,
        target: str,
        min_group_size: int = 5,
) -> dict:
    """
    Test de Kruskal-Wallis entre une variable numérique et une target qualitative.

    - Les NaN sont retirés temporairement.
    - Adapté aux distributions non normales.
    - Retourne la p-value et l'epsilon carré.
    """

    data = (
        df.select([col, target])
        .drop_nulls()
        .to_pandas()
    )

    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col, target])

    groupes = [
        group[col].values
        for _, group in data.groupby(target)
        if len(group) >= min_group_size
    ]

    if len(groupes) < 2:
        return {
            "variable": col,
            "type_variable": "numerique",
            "test": "kruskal",
            "n": len(data),
            "nb_groupes": len(groupes),
            "statistique": np.nan,
            "p_value": np.nan,
            "taille_effet": np.nan,
            "effet": "epsilon_squared",
            "interpretation": "non_testable",
        }

    h_stat, p_value = kruskal(*groupes)

    n = len(data)
    k = len(groupes)

    epsilon_squared = (h_stat - k + 1) / (n - k) if n > k else np.nan
    epsilon_squared = max(0, epsilon_squared)

    if epsilon_squared < 0.01:
        interpretation = "tres_faible"
    elif epsilon_squared < 0.06:
        interpretation = "faible"
    elif epsilon_squared < 0.14:
        interpretation = "moderee"
    else:
        interpretation = "forte"

    return {
        "variable": col,
        "type_variable": "numerique",
        "test": "kruskal",
        "n": int(n),
        "nb_groupes": k,
        "statistique": round(h_stat, 4),
        "p_value": p_value,
        "taille_effet": round(epsilon_squared, 4),
        "effet": "epsilon_squared",
        "interpretation": interpretation,
    }

def test_missing_vs_target(
    df: pl.DataFrame,
    col: str,
    target: str,
) -> dict:
    """
    Teste si le fait qu'une variable soit manquante est associé à la target.
    """

    if df.select(pl.col(col).null_count()).item() == 0:
        return {
            "variable": f"{col}_missing",
            "type_variable": "missing_indicator",
            "test": "chi2",
            "n": df.height,
            "nb_modalites": 1,
            "statistique": np.nan,
            "p_value": np.nan,
            "taille_effet": np.nan,
            "effet": "cramers_v",
            "interpretation": "aucun_missing",
        }

    temp_col = f"{col}_missing_temp"

    df_temp = df.with_columns(
        pl.col(col)
        .is_null()
        .cast(pl.Utf8)
        .alias(temp_col)
    )

    result = test_chi2_cramers_v(
        df=df_temp,
        col=temp_col,
        target=target,
    )

    result["variable"] = f"{col}_missing"

    return result

def tester_variables_avec_target(
    df: pl.DataFrame,
    target: str,
    cat_cols: list[str],
    numeric_cols: list[str],
    tester_missing: bool = True,
) -> pl.DataFrame:
    """
    Applique les tests statistiques entre les variables explicatives et la target.

    - Qualitatives : Chi² + V de Cramer
    - Numériques : Kruskal-Wallis + epsilon²
    - Missing : Chi² entre indicatrice temporaire de missing et target
    """

    resultats = []

    for col in cat_cols:
        if col in df.columns and col != target:
            resultats.append(
                test_chi2_cramers_v(df, col, target)
            )

            if tester_missing:
                resultats.append(
                    test_missing_vs_target(df, col, target)
                )

    for col in numeric_cols:
        if col in df.columns and col != target:
            resultats.append(
                test_kruskal_epsilon(df, col, target)
            )

            if tester_missing:
                resultats.append(
                    test_missing_vs_target(df, col, target)
                )

    return (
        pl.DataFrame(resultats)
        .sort("taille_effet", descending=True, nulls_last=True)
    )