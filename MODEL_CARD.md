# Fiche modèle — consommation énergétique

## Finalité

Estimer la consommation annuelle normalisée d'un bâtiment non résidentiel à partir de caractéristiques structurelles et d'usage. Le démonstrateur permet d'explorer la sensibilité du modèle et l'apport de l'ENERGY STAR Score.

## Modèle et données

- Random Forest compact de 300 arbres pour accélérer le chargement du démonstrateur.
- Entraînement sur la transformation `log1p` de `SiteEnergyUseWN(kBtu)`.
- Données publiques Seattle 2016 : 1 535 bâtiments après nettoyage.
- Découpage 70/30 stratifié par type de bâtiment, avec 461 observations de test.

## Performances

Sur les 461 bâtiments du jeu de test, la variante compacte avec ENERGY STAR Score obtient un R² de 0,515, une MAE de 3,30 millions de kBtu, une RMSE de 14,44 millions de kBtu et une erreur absolue médiane de 0,84 million de kBtu. Les valeurs exactes sont conservées dans `artifacts/demo_metadata.json` et affichées dans l'application.

Le modèle d'étude à 1 500 arbres reste la référence méthodologique documentée : R² de 0,502, MAE de 3,33 millions de kBtu, RMSE de 14,64 millions de kBtu et erreur absolue médiane de 0,85 million de kBtu.

## Limites

- Données limitées à Seattle et à l'année 2016.
- Erreurs très dispersées, particulièrement pour certains grands consommateurs.
- Absence de validation temporelle ou géographique.
- Les variations interactives ne prouvent pas un effet causal.

## Usage exclu

Le modèle ne doit pas être utilisé pour produire un diagnostic réglementaire, facturer une consommation, certifier un bâtiment ou prendre seul une décision d'investissement.
