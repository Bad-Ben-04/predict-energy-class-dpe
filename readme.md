# Prédiction de la Classe Énergétique d’un Logement — DPE

Projet de machine learning visant à prédire l’étiquette énergétique DPE d’un logement, de **A à G**, à partir des données publiques de l’ADEME.

---

## Objectif

Construire un modèle de classification capable de prédire la variable cible :

```txt
etiquette_dpe
```

Classes possibles :

```txt
A, B, C, D, E, F, G
```

Les variables explicatives retenues décrivent principalement :

- les caractéristiques générales du logement ;
- la localisation ;
- la période ou l’année de construction ;
- l’isolation ;
- le système de chauffage ;
- l’eau chaude sanitaire ;
- la ventilation ;
- les équipements énergétiques.

Les variables directement liées au calcul final du DPE sont exclues afin d’éviter le **data leakage** :

- consommations énergétiques ;
- émissions de gaz à effet de serre ;
- coûts ;
- besoins ;
- déperditions.

---

## Source des données

Les données proviennent de l’ADEME :

```txt
DPE Logements existants depuis juillet 2021
```

Source officielle :

```txt
https://data.ademe.fr/datasets/dpe03existant
```

Les données sont récupérées via l’API publique ADEME.

---

## Périmètre des données

Périmètre retenu pour ce projet :

```txt
Région : Hauts-de-France
Période : du 2025-01-01 jusqu’à la date de collecte
```

Les données brutes doivent être placées dans le dossier :

```txt
resources/
```

Exemple de fichier attendu :

```txt
resources/dpe_hauts_de_france_2025-01-01_to_2026-06-04.csv
```

---

## Prérequis

Version recommandée :

```txt
Python >= 3.13
```

Vérifier l’installation de Python :

```bash
python --version
```

Vérifier l’installation de pip :

```bash
pip --version
```

---

## Installation

Cloner le projet :

```bash
git clone https://github.com/Bad-Ben-04/predict-energy-class-dpe.git
```

Se placer dans le dossier du projet :

```bash
cd predict-energy-class-dpe
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l’environnement virtuel.

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous macOS ou Linux :

```bash
source .venv/bin/activate
```

Mettre à jour pip :

```bash
python -m pip install --upgrade pip
```

Installer les dépendances du projet :

```bash
pip install -r requirements-lock.txt
```

---

## Installation complète en une seule séquence

### Windows

```bash
git clone https://github.com/Bad-Ben-04/predict-energy-class-dpe.git
cd predict-energy-class-dpe
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

### macOS / Linux

```bash
git clone https://github.com/Bad-Ben-04/predict-energy-class-dpe.git
cd predict-energy-class-dpe
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

---

## Vérification de l’installation

Lancer Python :

```bash
python
```

Tester les imports principaux :

```python
import pandas as pd
import numpy as np
import sklearn
import requests

print("Installation OK")
```

Quitter Python :

```python
exit()
```

---

## Jupyter Notebook

Installer le kernel Jupyter associé à l’environnement virtuel :

```bash
python -m ipykernel install --user --name dpe-project --display-name "Python - DPE Project"
```

Lancer Jupyter :

```bash
jupyter notebook
```

ou :

```bash
jupyter lab
```

Sélectionner le kernel suivant dans Jupyter :

```txt
Python - DPE Project
```

## Rôle des fichiers d’installation

### `requirements-lock.txt`

Fichier principal pour installer l’environnement du projet.

Il contient les versions exactes des packages utilisés.

Commande à utiliser :

```bash
pip install -r requirements-lock.txt
```

Ce fichier permet à tous les membres du projet de travailler avec le même environnement Python.

### `pyproject.toml`

Fichier de configuration du projet.

Il décrit le projet et ses dépendances principales.

Il n’est pas utilisé en priorité pour reproduire l’environnement exact.

---

## Mise à jour des dépendances

Lorsqu’une nouvelle dépendance est ajoutée au projet :

```bash
pip install nom_du_package
```

Mettre ensuite à jour le fichier de verrouillage :

```bash
pip freeze > requirements-lock.txt
```

Commit les modifications :

```bash
git add requirements-lock.txt pyproject.toml
git commit -m "Update project dependencies"
```

## Problèmes fréquents

### L’environnement virtuel n’est pas activé

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous macOS ou Linux :

```bash
source .venv/bin/activate
```

---

### Erreur d’import d’un package

Réinstaller les dépendances :

```bash
pip install -r requirements-lock.txt
```

---

### Jupyter ne détecte pas le bon environnement

Réinstaller le kernel :

```bash
python -m ipykernel install --user --name dpe-project --display-name "Python - DPE Project"
```

Relancer ensuite Jupyter.

---

### Environnement différent entre deux machines

Supprimer l’environnement virtuel local, puis le recréer.

Sous Windows :

```bash
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

Sous macOS ou Linux :

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

---

## Désactivation de l’environnement virtuel

```bash
deactivate
```