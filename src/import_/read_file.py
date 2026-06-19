import pandas as pd
import polars as pl
import polars.selectors as cs
from IPython.display import display, Markdown

from collections import Counter

def read_file(path, target):
    display(Markdown("### Import des données"))

    df = pl.read_csv(path, low_memory=False)

    num_cols = list(df.select(cs.numeric()).columns)
    cat_cols = list(df.select(~cs.numeric()).columns)

    display(Markdown(
        f"Le jeu de données contient **{df.shape[0]} lignes** et **{df.shape[1]} colonnes**."
    ))

    df = reduce_memory_data(df)

    display(Markdown("### Aperçu des données importées"))

    display(df.head(8))

    display(Markdown("### Que vaut notre target ?"))

    display(
        df[target]
        .value_counts(normalize=True)
        .with_columns(
            (pl.col("proportion") * 100).round(2).alias("Pourcentage")
        )
        .sort(target)
        .drop("proportion")
    )

    return df, num_cols, cat_cols

def df_dtypes(df: pl.DataFrame):
    """Récucpère le Dataframe et affiche les difféérents types du df"""

    display(Markdown("### Les différents types du df"))
    display(pl.DataFrame(
        {
            "Variables": list(df.schema.keys()),
            "Types": [str(dtype) for dtype in df.schema.values()]
        }
    ))

    type_counts = Counter(df.dtypes)
    total = len(df.dtypes)

    df_types_percent = pl.DataFrame({
        "Type": list(type_counts.keys()),
        "Nombre": list(type_counts.values()),
    }).with_columns(
        ((pl.col("Nombre") / total) * 100).round(2).alias("Pourcentage")
    )

    display(Markdown("### Catégorisation des types du df"))
    display(df_types_percent)


def reduce_memory_data(df: pl.DataFrame) -> pl.DataFrame:
    display(Markdown("### Réduction de la taille du DataFrame"))

    start_size = df.estimated_size("mb")
    display(Markdown(f"Taille du DataFrame avant réduction : **{start_size:.2f} MB**"))

    cast_dict = {}

    int_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
    ]

    float_types = [pl.Float32, pl.Float64]

    num_cols = df.select(cs.numeric()).columns

    for col in num_cols:
        col_dtype = df[col].dtype
        min_col = df[col].min()
        max_col = df[col].max()

        # Si la colonne est vide ou uniquement composée de null
        if min_col is None or max_col is None:
            continue

        if col_dtype in int_types:

            if min_col >= 0 and max_col <= 255:
                cast_dict[col] = pl.UInt8

            elif min_col >= -128 and max_col <= 127:
                cast_dict[col] = pl.Int8

            elif min_col >= 0 and max_col <= 65_535:
                cast_dict[col] = pl.UInt16

            elif min_col >= -32_768 and max_col <= 32_767:
                cast_dict[col] = pl.Int16

            elif min_col >= 0 and max_col <= 4_294_967_295:
                cast_dict[col] = pl.UInt32

            elif min_col >= -2_147_483_648 and max_col <= 2_147_483_647:
                cast_dict[col] = pl.Int32

        elif col_dtype in float_types:

            # On réduit seulement les Float64 en Float32
            if col_dtype == pl.Float64:
                cast_dict[col] = pl.Float32

    df = df.cast(cast_dict)

    end_size = df.estimated_size("mb")
    reduction = 100 * (start_size - end_size) / start_size

    display(Markdown(f"Taille du DataFrame après réduction : **{end_size:.2f} MB**"))
    display(Markdown(f"Réduction mémoire : **{reduction:.2f}%**"))

    return df



