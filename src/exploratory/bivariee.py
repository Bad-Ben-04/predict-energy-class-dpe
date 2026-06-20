import polars as pl

def taux_energivore_par_modalite(
    df: pl.DataFrame,
    col: str,
    target: str = "",
    min_effectif: int = 30,
) -> pl.DataFrame:
    """
    Calcule le taux de logements énergivores pour chaque modalité.
    """

    df_tmp = df.select([col, target]).with_columns(
        [
            pl.col(col).cast(pl.Utf8).fill_null("Non renseigné"),
            pl.col(target).cast(pl.Utf8),
        ]
    )

    total_modalite = (
        df_tmp.group_by(col)
        .len()
        .rename({"len": "effectif_modalite"})
    )

    nb_energivore = (
        df_tmp.filter(pl.col(target) == "energivore")
        .group_by(col)
        .len()
        .rename({"len": "effectif_energivore"})
    )

    return (
        total_modalite
        .join(nb_energivore, on=col, how="left")
        .with_columns(
            pl.col("effectif_energivore").fill_null(0)
        )
        .with_columns(
            (pl.col("effectif_energivore") / pl.col("effectif_modalite") * 100)
            .round(2)
            .alias("taux_energivore_%")
        )
        .filter(pl.col("effectif_modalite") >= min_effectif)
        .sort("taux_energivore_%", descending=True)
    )


#============================================================================================================
#==================================== QUALI - QUALI =========================================================
#============================================================================================================

def tableau_quali_quali(
    df: pl.DataFrame,
    col1: str,
    col2: str,
) -> pl.DataFrame:
    """
    Tableau croisé entre deux variables qualitatives.
    Calcule les effectifs et les pourcentages par modalité de col1.
    """

    df_tmp = df.select([col1, col2]).with_columns(
        [
            pl.col(col1).cast(pl.Utf8).fill_null("Non renseigné"),
            pl.col(col2).cast(pl.Utf8).fill_null("Non renseigné"),
        ]
    )

    total_col1 = (
        df_tmp.group_by(col1)
        .len()
        .rename({"len": "effectif_col1"})
    )

    return (
        df_tmp.group_by([col1, col2])
        .len()
        .rename({"len": "effectif"})
        .join(total_col1, on=col1, how="left")
        .with_columns(
            (pl.col("effectif") / pl.col("effectif_col1") * 100)
            .round(2)
            .alias("pourcentage_dans_col1")
        )
        .sort([col1, "effectif"], descending=[False, True])
    )