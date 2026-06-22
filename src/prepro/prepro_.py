from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

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
    n_clusters: int = 3,
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


def imputer_apres_selection(
    df: pl.DataFrame,
    ordinal_cols: list[str],
    nominal_cols: list[str],
    numeric_cols: list[str],
    numeric_cols_missing: list[str],
    ordinal_modes: dict[str, str],
    valeur_nominale: str = "Non renseigné",
    valeur_num: float = -9999,
    add_ordinal_missing_indicator: bool = True,
) -> pl.DataFrame:
    """
    Imputation finale pour la modélisation.

    Ordinales :
    - remplacement des valeurs manquantes par le mode calculé sur le train
    - optionnel : création d'une indicatrice de valeur manquante

    Nominales :
    - remplacement des valeurs manquantes par "Non renseigné"

    Numériques :
    - remplacement des valeurs manquantes par une valeur numérique
    - indicatrice uniquement pour les colonnes qui avaient des valeurs manquantes dans le train
    """

    expressions = []

    # 1. Variables ordinales : imputation par le mode du train
    for col in ordinal_cols:
        if col not in ordinal_modes:
            raise ValueError(
                f"Aucun mode fourni pour la colonne ordinale '{col}'. "
                "Calcule les modes avec calculer_modes_train() sur le train."
            )

        if add_ordinal_missing_indicator:
            expressions.append(
                pl.col(col)
                .is_null()
                .cast(pl.Int8)
                .alias(f"{col}_missing")
            )

        expressions.append(
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null(ordinal_modes[col])
            .alias(col)
        )

    # 2. Variables nominales : imputation par "Non renseigné"
    for col in nominal_cols:
        expressions.append(
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null(valeur_nominale)
            .alias(col)
        )

    # 3. Variables numériques : imputation + indicatrice
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


def separer_colonnes_pipeline(
    df: pl.DataFrame,
    features: list[str],
):
    """
    Sépare les variables déjà imputées en :
    - cat_cols : variables catégorielles texte
    - binary_cols : variables binaires 0/1, booléennes ou indicatrices missing
    - numeric_cols : variables numériques non binaires
    """

    cat_cols = []
    binary_cols = []
    numeric_cols = []

    numeric_dtypes = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    ]

    for col in features:
        dtype = df[col].dtype

        if dtype in [pl.Utf8, pl.String, pl.Categorical]:
            cat_cols.append(col)

        elif dtype == pl.Boolean:
            binary_cols.append(col)

        elif dtype in numeric_dtypes:
            nb_modalites = df[col].drop_nulls().n_unique()

            if nb_modalites <= 2:
                binary_cols.append(col)
            else:
                numeric_cols.append(col)

    return cat_cols, binary_cols, numeric_cols

def dendrogramme_modalites_vs_target(
    df,
    col,
    target,
    min_effectif=100,
    method="ward",
    metric="euclidean",
    figsize=(12, 6)
):
    """
    Construit un dendrogramme des modalités d'une variable qualitative
    à partir de leur profil de répartition selon la target.

    Exemple :
    - col = "periode_construction"
    - target = "classe_dpe_4" ou la cible regroupée en 3 classes
    """

    data = (
        df.select([col, target])
        .drop_nulls()
        .to_pandas()
    )

    counts = pd.crosstab(data[col], data[target])

    # On retire les modalités trop rares
    effectifs = counts.sum(axis=1)
    counts = counts.loc[effectifs >= min_effectif]

    # Profil en proportions par modalité
    profils = counts.div(counts.sum(axis=1), axis=0)

    # Distance entre profils
    distances = pdist(profils.values, metric=metric)

    # Clustering hiérarchique
    Z = linkage(distances, method=method)

    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=profils.index.astype(str).tolist(),
        leaf_rotation=45,
        leaf_font_size=10
    )
    plt.title(f"Dendrogramme des modalités de {col}")
    plt.ylabel("Distance entre profils DPE")
    plt.tight_layout()
    plt.show()

    return profils, Z

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer, MissingIndicator


def construire_preprocessor(
    ordinal_cols,
    nominal_cols,
    numeric_cols,
    binary_cols,
    ordinal_categories,
    add_ordinal_missing_indicator=True,
):
    transformers = []

    if ordinal_cols:
        ordinal_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )
                ),
            ]
        )

        transformers.append(
            ("ordinal", ordinal_pipeline, ordinal_cols)
        )

        if add_ordinal_missing_indicator:
            transformers.append(
                (
                    "ordinal_missing",
                    MissingIndicator(error_on_new=False),
                    ordinal_cols,
                )
            )

    if nominal_cols:
        nominal_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Non renseigné"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                ),
            ]
        )

        transformers.append(
            ("nominal", nominal_pipeline, nominal_cols)
        )

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True
                    )
                ),
                (
                    "scaler",
                    RobustScaler()
                ),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if binary_cols:
        transformers.append(
            ("binary", "passthrough", binary_cols)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )