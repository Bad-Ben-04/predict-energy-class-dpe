import os
import time
import requests
import pandas as pd
from datetime import date

BASE_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
REGIONS = {
    "hauts_de_france": "32"
}

OUTPUT_DATA = "resources"

BASE_COLUMNS = [
    "numero_dpe",
    "date_etablissement_dpe",
    "etiquette_dpe",
    "code_region_ban",
    "code_departement_ban",
    "code_postal_ban",
    "code_insee_ban",
    "nom_commune_ban",
]

COLUMNS_TO_KEEP = [
    # Cible
    "etiquette_dpe",

    # Identifiants / filtres / localisation
    "numero_dpe",
    "date_etablissement_dpe",
    "code_region_ban",
    "code_departement_ban",
    "code_postal_ban",
    "code_insee_ban",
    "nom_commune_ban",
    "zone_climatique",
    "classe_altitude",

    # Caractéristiques générales du logement / bâtiment
    "type_batiment",
    "methode_application_dpe",
    "periode_construction",
    "annee_construction",
    "surface_habitable_immeuble",
    "nombre_niveau_immeuble",
    "nombre_appartement",
    "numero_etage_appartement",
    "hauteur_sous_plafond",
    "classe_inertie_batiment",

    # Isolation / enveloppe
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
    "qualite_isolation_plancher_bas",
    "qualite_isolation_plancher_haut_comble_perdu",

    # Chauffage
    "type_installation_chauffage",
    "type_installation_chauffage_n1",
    "configuration_installation_chauffage_n1",
    "type_generateur_chauffage_principal",
    "type_generateur_n1_installation_n1",
    "type_energie_principale_chauffage",
    "type_energie_generateur_n1_installation_n1",
    "type_emetteur_installation_chauffage_n1",
    "usage_generateur_n1_installation_n1",

    # Eau chaude sanitaire ECS
    "type_installation_ecs",
    "type_installation_ecs_n1",
    "configuration_installation_ecs_n1",
    "type_generateur_n1_ecs_n1",
    "type_generateur_chauffage_principal_ecs",
    "type_energie_principale_ecs",
    "type_energie_generateur_n1_ecs_n1",
    "usage_generateur_n1_ecs_n1",
    "volume_stockage_generateur_n1_ecs_n1",

    # Ventilation
    "ventilation_posterieure_2012",

    # Production renouvelable / solaire
    "type_installation_solaire_n1",
    "production_electricite_pv_kwhep_par_an",
]

def fetch_region_to_csv(
    url,
    region_name,
    region_code,
    start_date="2024-08-01",
    end_date= "2025-12-31",
    output_dir="resources",
    page_size=10000,
    pause=0.2,
):
    if end_date is None:
        end_date = date.today().isoformat()

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"dpe_{region_name}_{start_date}_to_{end_date}.csv"
    )

    assert len(COLUMNS_TO_KEEP) == len(set(COLUMNS_TO_KEEP)), "Colonnes dupliquées dans COLUMNS_TO_KEEP"

    params = {
        "format": "json",
        "q_mode": "simple",
        "qs": (
            f'code_region_ban:"{region_code}" '
            f'AND date_etablissement_dpe:[{start_date} TO {end_date}]'
        ),
        "size": page_size,
        "select": ",".join(COLUMNS_TO_KEEP)
    }

    total_rows = 0
    first_write = True
    page = 1

    while url:
        response = requests.get(
            url,
            params=params if page == 1 else None,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        rows = data.get("results", [])

        if not rows:
            break

        df_page = pd.json_normalize(rows)

        # Force les mêmes colonnes, dans le même ordre, à chaque page
        df_page = df_page.reindex(columns=COLUMNS_TO_KEEP)

        df_page.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8"
        )

        total_rows += len(df_page)

        print(
            f"{region_name} | page {page} | "
            f"{len(df_page)} lignes | total : {total_rows}"
        )

        first_write = False
        url = data.get("next")
        params = None
        page += 1

        time.sleep(pause)

    print(f"Terminé : {output_path} avec {total_rows} lignes")
    return output_path


def main():
    start_date = "2024-08-01"
    end_date = "2025-12-31"

    for region_name, region_code in REGIONS.items():
        fetch_region_to_csv(
            url=BASE_URL,
            region_name=region_name,
            region_code=region_code,
            start_date=start_date,
            end_date=end_date,
            output_dir=OUTPUT_DATA,
            page_size=10000,
            pause=0.05
        )


if __name__ == "__main__":
    main()