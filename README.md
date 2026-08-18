# Démonstrateur — consommation énergétique des bâtiments

Application Streamlit construite à partir du projet OpenClassrooms consacré à la prédiction de la consommation des bâtiments non résidentiels de Seattle.

**[Ouvrir la démonstration publique](https://conso-batiments-seattle.streamlit.app/)**

Le démonstrateur utilise une variante compacte de 300 arbres, validée séparément du modèle d'étude à 1 500 arbres, afin de réduire le poids des artefacts et le temps de chargement.

## Fonctionnalités

- exploration en lecture seule de trois bâtiments anonymisés du jeu de test ;
- comparaison de leurs prédictions à la consommation réellement observée ;
- mode de simulation séparé pour modifier les caractéristiques d'un bâtiment fictif ;
- comparaison des estimations avec et sans ENERGY STAR Score ;
- conversion des résultats en MWh et comparaison à la médiane du jeu de test en mode simulation ;
- affichage du protocole de validation et des limites d'usage.

## Lancer l'application en local

Avec Python 3.11, depuis ce dossier :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Sous macOS ou Linux, remplacez l'activation par `source .venv/bin/activate`.

## Reconstruire les artefacts

```powershell
python build_artifacts.py
```

La reconstruction nécessite les données source du projet, qui ne sont pas publiées dans ce dépôt. Les artefacts nécessaires à la démonstration sont déjà versionnés dans `artifacts/`.

## Usage responsable

Cette application est une démonstration éducative. Les estimations ne remplacent ni une mesure énergétique, ni un diagnostic réglementaire, ni une expertise de terrain.
