# Weigh LifeStyle — Prédiction du poids (LightGBM & RandomForest)

Projet de régression qui lie hygiène de vie et poids corporel. Les notebooks fournissent l'EDA initiale tandis que la pipeline automatisée gère nettoyage, splits, tuning Optuna, entraînement et évaluation (métriques + SHAP).

## Objectifs
- Obtenir un pipeline reproductible sans fuite de données du test.
pip install -r requirements-dev.txt
- Pinner l'entraînement LightGBM avec un jeu d'hyperparamètres consigné.
- Conserver tous les artefacts (données dérivées, modèles, graphiques) dans le dépôt pour audit.

## Flux de données
1. `data/dataset.csv` → notebooks → `data/dataset_cleaned.csv`.
2. `dataset_cleaned.csv` → script de nettoyage → `data/dataset_cleaned_final.(csv|parquet)`.
3. `dataset_cleaned_final` → préparation → `data/dataset_splits_encoded.(csv|parquet)` + mappings.
4. Splits encodés → feature importance, hyperparameter search, entraînement et évaluation.

Chaque étape est détaillée dans `documentation/methodologie.txt`.

## Structure du dépôt

```
data/                         # Données brutes, intermédiaires et finales
notebooks/                    # EDA univariée et bivariée
plots/                        # Visualisations (distributions, corrélations, SHAP, permutation)
results/                      # Hyperparamètres, métriques et modèles sauvegardés
src/
  data_cleaning/              # Normalisation colonnes + binarisation exercise
  data_preparation/           # Shuffle, split train/val/test, encodage LabelEncoder
  feature_engineering/        # Importance par permutation
  hyperparameters_optimization/ # Optuna (LightGBM & RandomForest)
  training/                   # Entraînement LightGBM + RandomForest
  eval/                       # Évaluation test + SHAP
  utils.py                    # Fonctions partagées (chargement, encodage, validations)
main_global.py                # Orchestrateur end-to-end
documentation/methodologie.txt # Méthodologie détaillée
documentation/Lightgbm_results.md # Synthèse JSON LightGBM prête à lire
documentation/rf_results.md      # Synthèse JSON RandomForest prête à lire

```

## Installation
Prérequis : Python 3.10+.

```
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# Optionnel pour tests/typing :

```

Les dépendances couvrent LightGBM, Optuna, SHAP, Matplotlib/Seaborn (EDA & plots) et scikit-learn.

## Notebooks d'EDA
- `notebooks/analyse_univariee.ipynb` : distribs, valeurs extrêmes, export `dataset_cleaned.csv`.
- `notebooks/analyse_bivariee_multivariee.ipynb` : corrélations, scatterplots, profils catégoriels.

Rerun lorsque la structure des données évolue afin de maintenir les exports cohérents.

## Pipeline CLI
Les étapes peuvent être lancées individuellement ou via l'orchestrateur global.

| Étape | Commande | Sorties clés |
|-------|----------|--------------|
| Nettoyage | `python -m src.data_cleaning.main` | `data/dataset_cleaned_final.(csv|parquet)`, plot de distribution |
| Préparation | `python -m src.data_preparation.main` | Splits encodés + mappings LabelEncoder |
| Hyperparamètres | `python -m src.hyperparameters_optimization.main --models both` | `results/best_lightgbm_params.json`, `results/best_random_forest_params.json` |
| Feature importance | `python -m src.feature_engineering.main` | JSON + graphique de permutation |
| Entraînement | `python -m src.training.main --models both` | `results/models/*.joblib` + métriques validation |
| Évaluation | `python -m src.eval.main --models both` | `results/eval/*.json`, rapports SHAP |
| End-to-end | `python -m src.main_global` | Enchaînement complet |

Le tuning Optuna se fait sur un split train/val fixe (pas de CV) afin de limiter le temps de calcul tout en gardant le test hermétique.

## Journalisation
- `src.utils.get_logger` configure un logger unique (format homogène, sortie standard) partagé par tous les modules CLI.
- Remplacer tout `print` applicatif par `LOGGER = get_logger(__name__)` puis `LOGGER.info(...)`/`LOGGER.warning(...)` selon le niveau souhaité.
- Les logs d'erreur utilisent `LOGGER.error` afin d'être capturés par les pipelines externes tout en conservant la trace complète.
- Aucun test n'instancie ce logger pour éviter les dépendances implicites ; seules les commandes utilisateur y font appel.

## Résultats clés

### Métriques sur le jeu de test
- **LightGBM** :
  - MSE : 1.34 | RMSE : 1.16 | MAE : 0.74 | R² : 0.9970
  - Hyperparamètres optimaux dans `results/best_lightgbm_params.json`
  - Documentation détaillée : `documentation/Lightgbm_results.md`
  
- **Random Forest** :
  - MSE : 0.67 | RMSE : 0.82 | MAE : 0.52 | R² : 0.9985
  - Hyperparamètres optimaux dans `results/best_random_forest_params.json`
  - Documentation détaillée : `documentation/rf_results.md`

- **Features les plus importantes** (SHAP) : `water-intake-(liters)`, `cholesterol-mg`, `age`, `session-duration-(hours)`.
- **Visualisations** : métriques + SHAP enregistrés sous `results/eval/` et `plots/shape/`.

## Reproductibilité
- Seeds centralisés dans `src/constants.py` (`DEFAULT_RANDOM_STATE`).
- Encodages catégoriels persistés (`data/encoders_mappings.*`).
- Splits et artefacts écrits dans des chemins stables pour faciliter la ré-exécution.
- Les tests unitaires (`pytest`) s'appuient sur des données simulées (mock) pour ne pas dépendre des CSV réels.

## Assistance LLM
Ce projet a été développé avec l'assistance d'un LLM **uniquement pour l'écriture des lignes de code**. La méthodologie, l'architecture et les décisions techniques ont été conçues par l'auteur. Pour plus de détails sur l'utilisation de l'IA dans ce projet, consultez `documentation/LLM_assistance_methodologie.txt`.

## Pistes d'amélioration
- Étendre le tuning Optuna à une validation croisée pour les scénarios serveur.
- Ajouter un rapport de monitoring (MLflow/Weights & Biases) pour suivre les runs.
- Expérimenter d'autres modèles gradient boosting (CatBoost, XGBoost) dans la même infrastructure.

## Licence
À définir par le propriétaire du dépôt.
