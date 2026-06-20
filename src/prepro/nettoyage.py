import polars as pl


def nettoyage_metier(df: pl.DataFrame) -> pl.DataFrame:
    """
    Nettoyage métier.

    Objectifs :
    1. Corriger hauteur_sous_plafond si les valeurs semblent être en centimètres.
       Aucune suppression n'est faite ici sur cette variable.
    2. Créer une variable has_PV pour la production photovoltaïque.
    3. Corriger numero_etage_appartement :
       - appartement : valeur conservée telle quelle, y compris null
       - maison : valeur ramenée à 0
       - immeuble : valeur ramenée à 0
    4. Identifier les volumes ECS incohérents selon le type de bâtiment.
    5. Supprimer uniquement les lignes avec volume ECS incohérent.
    """

    COL_HAUTEUR = "hauteur_sous_plafond"
    COL_PV = "production_electricite_pv_kwhep_par_an"
    COL_VOLUME = "volume_stockage_generateur_n1_ecs_n1"
    COL_TYPE_BATIMENT = "type_batiment"
    COL_ETAGE = "numero_etage_appartement"

    df = df.with_columns(
        pl.col(COL_TYPE_BATIMENT)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .alias("_type_batiment_lower")
    )

    # Correction hauteur sous plafond
    df = df.with_columns(
        pl.when(pl.col(COL_HAUTEUR) > 100)
        .then(pl.col(COL_HAUTEUR) / 100)
        .otherwise(pl.col(COL_HAUTEUR))
        .alias("hauteur_sous_plafond_corrigee")
    )

    df = df.filter(
        pl.col("hauteur_sous_plafond_corrigee").is_between(1.80, 6.00)
    )

    # Variable présence photovoltaïque
    df = df.with_columns(
        (pl.col(COL_PV).fill_null(0) > 0)
        .cast(pl.Int8)
        .alias("has_PV")
    )

    # Correction numero_etage_appartement
    df = df.with_columns(
        pl.when(
            pl.col("_type_batiment_lower").is_in(["maison", "immeuble"])
        )
        .then(0)
        .otherwise(pl.col(COL_ETAGE))
        .alias("numero_etage_appartement_corrige")
    )

    # Flag volume ECS incohérent
    df = df.with_columns(
        pl.when(pl.col(COL_VOLUME).is_null())
        .then(False)

        .when(pl.col(COL_VOLUME) < 0)
        .then(True)

        .when(
            pl.col("_type_batiment_lower").is_in(["maison", "appartement"])
            & (pl.col(COL_VOLUME) > 500)
        )
        .then(True)

        .when(
            (pl.col("_type_batiment_lower") == "immeuble")
            & (pl.col(COL_VOLUME) > 5000)
        )
        .then(True)

        .otherwise(False)
        .alias("outlier_volume_stockage_ecs")
    )

    # Filtrage uniquement sur le volume ECS
    df = df.filter(
        pl.col("outlier_volume_stockage_ecs") == False
    )

    df = df.drop(["_type_batiment_lower", "outlier_volume_stockage_ecs"])

    return df
