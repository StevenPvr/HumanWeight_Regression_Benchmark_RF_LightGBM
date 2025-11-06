# Random Forest Evaluation Results

## Source Artefacts
- `results/best_random_forest_params.json`
- `results/models/random_forest_metrics.json`
- `results/eval/random_forest_test_metrics.json`

Tous les chemins sont stockés relativement au dépôt (`results/...`, `data/...`).

## Hyperparameter Optimisation
```json
{
  "best_params": {
    "n_estimators": 1783,
    "max_depth": 26,
    "min_samples_split": 7,
    "min_samples_leaf": 3,
    "max_features": 0.3487586260895855,
    "bootstrap": false
  },
  "best_value": 1.258771129595547,
  "data_summary": {
    "train_rows": 16000
  }
}
```

## Training Metadata
```json
{
  "model_path": "results/models/random_forest.joblib",
  "parquet_path": "data/dataset_splits_encoded.parquet",
  "params_path": "results/best_random_forest_params.json",
  "target_column": "weight-(kg)",
  "random_state": 123,
  "training_rows": 16000,
  "best_params": {
    "n_estimators": 1497,
    "max_depth": 29,
    "min_samples_split": 9,
    "min_samples_leaf": 1,
    "max_features": 0.38157692468479576,
    "bootstrap": false
  }
}
```

## Test Set Evaluation
```json
{
  "model_path": "results/models/random_forest.joblib",
  "parquet_path": "data/dataset_splits_encoded.parquet",
  "target_column": "weight-(kg)",
  "metrics": {
    "mae": 0.578513247581413,
    "mse": 1.027267533610976,
    "rmse": 1.013542072935789,
    "r2": 0.9977305398225511,
    "mape": 0.007968937678335012,
    "median_ae": 0.39712520613357327,
    "explained_variance": 0.9977356759103729,
    "y_std": 21.275535405643144,
    "pred_std": 20.82104125996497,
    "residual_std": 1.012394533589175
  },
  "shap": null
}
```

> 🔍 Les scripts `training` et `eval` partagent la même logique de conversion de chemins que LightGBM. Si un calcul SHAP Random Forest est ajouté ultérieurement, il apparaîtra sous `plots/shape/rf/`.
