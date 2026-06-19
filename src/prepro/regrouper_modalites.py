import polars as pl

import polars as pl


def regrouper_modalites(
    df: pl.DataFrame,
    dico: dict,
    col: str,
    nouvelle_col: str | None = None,
    valeur_defaut: str = "Autre",
) -> pl.DataFrame:
    if col not in df.columns:
        raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    if nouvelle_col is None:
        nouvelle_col = f"{col}_regroupee"

    col_lower = (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip_chars()
    )

    expr = pl.when(pl.col(col).is_null()).then(pl.lit(None))

    for groupe, mots_cles in dico.items():
        condition = None

        for mot in mots_cles:
            current_condition = col_lower.str.contains(
                mot.lower(),
                literal=True
            )

            condition = (
                current_condition
                if condition is None
                else condition | current_condition
            )

        if condition is not None:
            expr = expr.when(condition).then(pl.lit(groupe))

    expr = expr.otherwise(pl.lit(valeur_defaut)).alias(nouvelle_col)

    return df.with_columns(expr)


def appliquer_regroupements_modalites(
    df: pl.DataFrame,
    modalites_groupings: dict,
    suffixe: str = "_regroupee",
    valeur_defaut: str = "Autre",
) -> pl.DataFrame:
    """
    Applique tous les regroupements de modalités définis dans MODALITES_GROUPINGS.
    Les colonnes originales sont conservées.
    Une nouvelle colonne est créée avec le suffixe '_regroupee'.
    """

    for col, dico in modalites_groupings.items():
        nouvelle_col = f"{col}{suffixe}"

        df = regrouper_modalites(
            df=df,
            dico=dico,
            col=col,
            nouvelle_col=nouvelle_col,
            valeur_defaut=valeur_defaut,
        )

    return df


def creer_target_dpe_4_classes(
    df: pl.DataFrame,
    target_col: str = "etiquette_dpe",
    new_col: str = "classe_dpe_4",
) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(target_col).is_in(["A", "B"]))
        .then(pl.lit("tres_performant"))

        .when(pl.col(target_col) == "C")
        .then(pl.lit("performant"))

        .when(pl.col(target_col) == "D")
        .then(pl.lit("intermediaire"))

        .when(pl.col(target_col).is_in(["E", "F", "G"]))
        .then(pl.lit("energivore"))

        .otherwise(pl.lit(None))
        .alias(new_col)
    )