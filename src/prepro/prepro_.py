from sklearn.model_selection import train_test_split
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def split_train_test_polars(
    df: pl.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df_idx = df.with_row_index("__row_id")

    row_ids = df_idx["__row_id"].to_numpy()
    y = df_idx[target].to_numpy()

    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = (
        df_idx
        .filter(pl.col("__row_id").is_in(train_ids))
        .drop("__row_id")
    )

    test_df = (
        df_idx
        .filter(pl.col("__row_id").is_in(test_ids))
        .drop("__row_id")
    )

    return train_df, test_df

def elbow_plot(X, k_min=2, k_max=8, random_state=42):
    """
    Calcule et affiche la courbe du coude pour aider à choisir
    le nombre de clusters k avec KMeans.
    """

    k_range = list(range(k_min, k_max + 1))
    inertias = []

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.plot(k_range, inertias, marker="o")

    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude")
    plt.xticks(k_range)
    plt.grid(alpha=0.3)

    plt.show()

    return k_range, inertias


def fit_kmeans_pv_train(
    train_df: pl.DataFrame,
    col: str = "",
    n_clusters: int = 4,
    random_state: int = 42,
):
    X_train_pos = (
        train_df
        .filter(pl.col(col) > 0)
        .select(col)
        .to_numpy()
    )

    X_train_pos_log = np.log1p(X_train_pos)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    kmeans.fit(X_train_pos_log)

    centres_log = kmeans.cluster_centers_.flatten()
    ordre_clusters = np.argsort(centres_log)

    noms_classes = [
        "pv_tres_faible",
        "pv_faible",
        "pv_moyenne",
        "pv_elevee",
    ]

    mapping_cluster = {
        int(cluster_id): noms_classes[rang]
        for rang, cluster_id in enumerate(ordre_clusters)
    }

    return kmeans, mapping_cluster

def appliquer_kmeans_pv(
    df: pl.DataFrame,
    kmeans: KMeans,
    mapping_cluster: dict,
    col: str = "",
    new_col: str = "production_pv_kmeans",
) -> pl.DataFrame:
    df_pos = df.filter(pl.col(col) > 0)
    df_zero = df.filter(pl.col(col) == 0)
    df_null = df.filter(pl.col(col).is_null())

    if df_pos.height > 0:
        X_pos = df_pos.select(col).to_numpy()
        X_pos_log = np.log1p(X_pos)

        labels = kmeans.predict(X_pos_log)

        classes_pos = [
            mapping_cluster[int(label)]
            for label in labels
        ]

        df_pos = df_pos.with_columns(
            pl.Series(new_col, classes_pos)
        )

    df_zero = df_zero.with_columns(
        pl.lit("aucune_production_pv").alias(new_col)
    )

    df_null = df_null.with_columns(
        pl.lit(None).alias(new_col)
    )

    return pl.concat(
        [df_zero, df_pos, df_null],
        how="diagonal"
    )

def colonnes_avec_missing(df: pl.DataFrame, cols: list[str]) -> list[str]:
    """
    Retourne uniquement les colonnes qui contiennent au moins une valeur manquante.
    """
    return [
        col for col in cols
        if df.select(pl.col(col).null_count()).item() > 0
    ]

def imputer_valeurs_manquantes(
    df: pl.DataFrame,
    cat_cols: list[str],
    numeric_cols: list[str],
    cat_cols_missing: list[str],
    numeric_cols_missing: list[str],
    valeur_cat: str = "Non renseigné",
    valeur_num: float = -9999,
) -> pl.DataFrame:
    """
    Impute les valeurs manquantes.
    Crée une colonne indicatrice uniquement pour les variables
    qui avaient réellement des valeurs manquantes.
    """

    expressions = []

    # Variables catégorielles
    for col in cat_cols:
        if col in cat_cols_missing:
            expressions.append(
                pl.col(col)
                .is_null()
                .cast(pl.Int8)
                .alias(f"{col}_missing")
            )

        expressions.append(
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null(valeur_cat)
            .alias(col)
        )

    # Variables numériques
    for col in numeric_cols:
        if col in numeric_cols_missing:
            expressions.append(
                pl.col(col)
                .is_null()
                .cast(pl.Int8)
                .alias(f"{col}_missing")
            )

        expressions.append(
            pl.col(col)
            .fill_null(valeur_num)
            .alias(col)
        )

    return df.with_columns(expressions)

