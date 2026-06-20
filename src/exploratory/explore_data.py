import polars as pl

from IPython.display import display, Markdown

def show_values_counts_columns(df: pl.DataFrame):
    """
    Affiche le nombre de modalités distinctes et le pourcentage
    de valeurs manquantes pour chaque variable.
    """

    result = pl.DataFrame({
        "Variable": df.columns,
        "Nombre de lignes": [df.height] * len(df.columns),
        "% valeurs manquantes": [
            round((df[col].null_count() / df.height) * 100, 2)
            for col in df.columns
        ],
        "Modalités distinctes": [
            df[col].drop_nulls().n_unique()
            for col in df.columns
        ],
    }).with_columns(
        (
            (pl.col("Modalités distinctes") / pl.col("Nombre de lignes")) * 100
        ).round(2).alias("% modalités distinctes")
    ).sort("% valeurs manquantes", descending=True)

    display(Markdown("### Modalités distinctes et valeurs manquantes par variable"))
    display(result)


def analyse_univariee_qualitative(df: pl.DataFrame, col: str):
    """
    Analyse univariée d'une variable qualitative :
    - nombre de modalités
    - pourcentage de valeurs manquantes
    - répartition des modalités
    """

    total = df.height
    missing_count = df[col].null_count()
    missing_percentage = round((missing_count / total) * 100, 2)

    display(Markdown(f"## Analyse univariée : `{col}`"))

    display(Markdown(f"""
        - Nombre de lignes : **{total}**
        - Nombre de modalités distinctes : **{df[col].drop_nulls().n_unique()}**
        - Pourcentage de valeurs manquantes : **{missing_percentage} %**
        """
        )
    )

    repartition = (
        df[col]
        .value_counts(normalize=True, sort=True)
        .with_columns(
            (pl.col("proportion") * 100).round(2).alias("Pourcentage")
        )
        .drop("proportion")
    )

    display(repartition)


def resume_modalites(df: pl.DataFrame, col: str) -> pl.DataFrame:
    total = df.height

    return (
        df.group_by(col)
        .len()
        .rename({"len": "effectif"})
        .sort("effectif", descending=True)
        .with_columns(
            [
                (pl.col("effectif") / total * 100)
                .round(2)
                .alias("pourcentage"),

                (pl.col("effectif").cum_sum() / total * 100)
                .round(2)
                .alias("pourcentage_cumule"),
            ]
        )
    )

### VARIABLES NUMERIQUES

def describe_variables_numeriques(
    df: pl.DataFrame,
    numeric_cols: list,
    seuil_iqr: float = 1.5,
    min_unique_outlier: int = 5,
):
    """Donne un resume statistique des variables numériques"""

    total_lignes = df.height

    resultats = []

    for col in numeric_cols:
        serie = df[col]
        null_count = serie.null_count()
        missing_pct = round(null_count / total_lignes * 100, 2)

        serie_non_null = serie.drop_nulls()
        nb_non_null = len(serie_non_null)
        nb_unique = serie_non_null.n_unique() if nb_non_null > 0 else 0

        if nb_non_null == 0:
            resultats.append({
                "colonne": col,
                "count": 0,
                "missing_count": null_count,
                "missing_%": missing_pct,
                "nb_unique": nb_unique,
                "mean": None,
                "std": None,
                "min": None,
                "q1": None,
                "median": None,
                "q3": None,
                "max": None,
                "iqr": None,
                "borne_basse": None,
                "borne_haute": None,
                "outlier_count": None,
                "outlier_%": None,
            })
            continue

        q1 = serie_non_null.quantile(0.25)
        q3 = serie_non_null.quantile(0.75)
        iqr = q3 - q1 if q1 is not None and q3 is not None else None

        if nb_unique < min_unique_outlier:
            outlier_count = None
            outlier_pct = None
            borne_basse = None
            borne_haute = None

        else:
            borne_basse = q1 - seuil_iqr * iqr
            borne_haute = q3 + seuil_iqr * iqr

            outlier_count = df.filter(
                (pl.col(col) < borne_basse) | (pl.col(col) > borne_haute)
            ).height

            outlier_pct = round(outlier_count / total_lignes * 100, 2)

        resultats.append({
            "colonne": col,
            "count": nb_non_null,
            "missing_count": null_count,
            "missing_%": missing_pct,
            "nb_unique": nb_unique,
            "mean": round(serie_non_null.mean(), 3) if serie_non_null.mean() is not None else None,
            "std": round(serie_non_null.std(), 3) if serie_non_null.std() is not None else None,
            "min": serie_non_null.min(),
            "q1": q1,
            "median": serie_non_null.median(),
            "q3": q3,
            "max": serie_non_null.max(),
            "iqr": iqr,
            "borne_basse": borne_basse,
            "borne_haute": borne_haute,
            "outlier_count": outlier_count,
            "outlier_%": outlier_pct,
        })

    resume = pl.DataFrame(resultats)

    colonnes_outliers = (
        resume
        .filter(pl.col("outlier_count") > 0)
        .select(["colonne", "outlier_count", "outlier_%", "borne_basse", "borne_haute"])
        .sort("outlier_%", descending=True)
    )

    cols_outliers = colonnes_outliers.get_column("colonne").to_list()

    display(Markdown("### Resume statistique des variables numériques"))
    display(resume)

    return cols_outliers


## TARGET

def resume_target_dpe(
    df: pl.DataFrame,
    classes_dpe: list,
    target_col: str = "etiquette_dpe",
) :
    total = df.height

    ordre_classes = pl.DataFrame({
        target_col: classes_dpe
    })

    repartition = (
        df.group_by(target_col)
        .len()
        .rename({"len": "effectif"})
    )

    resume = (
        ordre_classes
        .join(repartition, on=target_col, how="left")
        .with_columns(
            pl.col("effectif").fill_null(0).cast(pl.Int64)
        )
        .with_columns(
            (pl.col("effectif") / total * 100)
            .round(2)
            .alias("pourcentage")
        )
    )

    return resume

## REGROUPEMENT

def controle_regroupements_modalites(
    df: pl.DataFrame,
    colonnes_regroupees: list[str],
) :

    """
        Cette fonction vise à regrouper les modalités des variables numériques
        possédant un grand nimbre de modalités au départ
    """

    resultats = []

    total = df.height

    for col in colonnes_regroupees:
        nb_modalites = df[col].drop_nulls().n_unique()
        nb_null = df[col].null_count()

        nb_autre = (
            df.filter(pl.col(col) == "Autre").height
            if "Autre" in df[col].drop_nulls().unique().to_list()
            else 0
        )

        resultats.append(
            {
                "colonne": col,
                "nb_modalites": nb_modalites,
                "missing_count": nb_null,
                "missing_%": round(nb_null / total * 100, 2),
                "autre_count": nb_autre,
                "autre_%": round(nb_autre / total * 100, 2),
            }
        )

    return pl.DataFrame(resultats).sort("autre_%", descending=True)


def resume_production_pv_positive(
        df: pl.DataFrame,
        col: str = "production_electricite_pv_kwhep_par_an",
) -> pl.DataFrame:
    """
        Cette fonction renvoie un resume statistique de la variable "production_electricite_pv_kwhep_par_an"
        En se basant uniquement sur des valeurs > 0
    """

    df_pos = df.filter(pl.col(col) > 0)

    return df_pos.select(
        [
            pl.len().alias("nb_valeurs_positives"),
            pl.col(col).min().alias("min"),
            pl.col(col).quantile(0.25).alias("q25"),
            pl.col(col).quantile(0.50).alias("mediane"),
            pl.col(col).quantile(0.75).alias("q75"),
            pl.col(col).quantile(0.90).alias("q90"),
            pl.col(col).quantile(0.95).alias("q95"),
            pl.col(col).quantile(0.99).alias("q99"),
            pl.col(col).max().alias("max"),
        ]
    )

