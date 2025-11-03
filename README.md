# Weigh LifeStyle — Prédiction du poids (LightGBM & RandomForest)

Projet de régression qui relie hygiène de vie et poids corporel. Les notebooks fournissent l'EDA initiale tandis que la pipeline automatisée gère nettoyage, splits, tuning Optuna, entraînement et évaluation (métriques + SHAP).

## Objectifs
- Obtenir un pipeline reproductible sans fuite de données du test.
- Pinner l'entraînement LightGBM avec un jeu d'hyperparamètres consigné.
- Conserver tous les artefacts (données dérivées, modèles, graphiques) dans le dépôt pour audit.

## Configuration de l’environnement
Prérequis : Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# Dépendances supplémentaires pour les tests et linters
pip install -r requirements-dev.txt
```

> 💡 Les scripts CLI sont exécutables sans installation du paquet grâce à l’injection du `PROJECT_ROOT` dans `sys.path`. Aucune variable d’environnement spécifique n’est nécessaire.

## Conventions & respect des instructions
- Les règles globales (PEP 8, KISS, DRY, type hints…) sont décrites dans `AGENTS.md` à la racine. Toute modification de code ou de documentation doit en tenir compte.
- Les chemins sont dérivés de `src/constants.py` : utiliser `PROJECT_ROOT`, `DATA_DIR`, `RESULTS_DIR`, etc., au lieu de concaténations manuelles.
- Pour la journalisation, remplacer tout `print` par `get_logger(__name__)` depuis `src.utils`.

## Flux de travail recommandé
### 1. Préparer les données sources
- Déposer le CSV brut dans `data/dataset.csv`.
- (Optionnel) Mettre à jour les notebooks si de nouvelles colonnes apparaissent.

### 2. Exécuter les notebooks d’EDA
- `notebooks/analyse_univariee.ipynb` : distribs, valeurs extrêmes, export `data/dataset_cleaned.csv`.
- `notebooks/analyse_bivariee_multivariee.ipynb` : corrélations, scatterplots, profils catégoriels.
- Sauvegarder les visualisations dans `plots/` pour conserver l’historique.

### 3. Lancer les scripts CLI
| Étape | Commande | Entrées | Sorties clés |
|-------|----------|---------|--------------|
| Nettoyage | `python -m src.data_cleaning.main` | `data/dataset_cleaned.csv` | `data/dataset_cleaned_final.(csv|parquet)`, plot de distribution |
| Préparation | `python -m src.data_preparation.main` | Fichier nettoyé final | Splits encodés + mappings LabelEncoder |
| Hyperparamètres | `python -m src.hyperparameters_optimization.main --models both` | Splits encodés | `results/best_lightgbm_params.json`, `results/best_random_forest_params.json` |
| Feature importance | `python -m src.feature_engineering.main` | Splits encodés | JSON + graphique de permutation |
| Entraînement | `python -m src.training.main --models both` | Splits encodés + JSON d’hyperparamètres | `results/models/*.joblib` + métriques validation |
| Évaluation | `python -m src.eval.main --models both` | Splits encodés + modèles | `results/eval/*.json`, rapports SHAP |
| End-to-end | `python -m src.main_global` | Orchestrateur | Enchaînement complet |

Les chemins par défaut des arguments CLI proviennent tous de `src/constants.py`. Les artefacts écrits lors du run sont automatiquement convertis en chemins relatifs via `src.utils.to_project_relative_path`, ce qui facilite le versioning.

### 4. Tests et vérifications
- `pytest` : exécute les tests unitaires et d’intégration.
- `pytest -k e2e` : rejoue le test end-to-end à partir des fixtures mockées.
- `python -m src.main_global` : smoke test manuel sur les fichiers présents dans `data/`.

### 5. Gestion des résultats
- Modèles sauvegardés dans `results/models/` (LightGBM et RandomForest).
- Métriques finales dans `results/eval/` (JSON) et SHAP dans `plots/shape/`.
- Les mappings d’encodage et splits restent dans `data/` pour rejouer la pipeline.

Chaque étape est détaillée dans `documentation/methodologie.txt`.

## Gestion centralisée des chemins
- `src/constants.py` définit les répertoires (`DATA_DIR`, `RESULTS_DIR`, `PLOTS_DIR`, etc.) et les noms de fichiers par défaut.
- `src.utils.to_project_relative_path` garantit que les chemins stockés dans les JSON sont relatifs (ex. `results/models/lightgbm.joblib`).
- Les scripts vérifient/créent les dossiers parents nécessaires avant d’écrire un fichier.

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

Les dépendances couvrent LightGBM, Optuna, SHAP, Matplotlib/Seaborn (EDA & plots) et scikit-learn. Pour l’analyse statique, `requirements-dev.txt` fournit `black`, `ruff` et `pytest`.

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

## Changements récents
- Normalisation des chemins via `src.constants` et `to_project_relative_path` pour éviter tout chemin absolu dans les artefacts.
- Ajout d’options `--models` sur les CLI `training` et `eval` afin de sélectionner LightGBM, RandomForest ou les deux.
- Harmonisation des tests d’intégration (`pytest`) autour de jeux de données synthétiques.

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
