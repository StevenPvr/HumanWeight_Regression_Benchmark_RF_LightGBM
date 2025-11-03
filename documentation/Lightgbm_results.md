# LightGBM Evaluation Results

## Source Artefacts
- `results/best_lightgbm_params.json`
- `results/models/lightgbm_metrics.json`
- `results/eval/lightgbm_test_metrics.json`

Tous les chemins enregistrés sont désormais relatifs au projet grâce à `src.utils.to_project_relative_path`.

## Hyperparameter Optimisation
```json
{
  "best_params": {
    "n_estimators": 1863,
    "learning_rate": 0.013855642404654556,
    "num_leaves": 354,
    "max_depth": 14,
    "min_child_samples": 26,
    "subsample": 0.7276280751687296,
    "colsample_bytree": 0.5543816813470782,
    "reg_lambda": 19.192849668958537,
    "reg_alpha": 10.047676285972248
  },
  "best_value": 1.379405705594721,
  "val_metrics": {
    "mse": 1.379405705594721
  },
  "data_summary": {
    "train_rows": 11999,
    "val_rows": 4001,
    "train_val_rows": 16000
  }
}
```

## Training Metadata
```json
{
  "model_path": "results/models/lightgbm.joblib",
  "parquet_path": "data/dataset_splits_encoded.parquet",
  "params_path": "results/best_lightgbm_params.json",
  "target_column": "weight-(kg)",
  "random_state": 123,
  "validation": {
    "mse": 1.379405705594721
  },
  "training_rows": 11999,
  "validation_rows": 4001,
  "best_params": {
    "n_estimators": 1863,
    "learning_rate": 0.013855642404654556,
    "num_leaves": 354,
    "max_depth": 14,
    "min_child_samples": 26,
    "subsample": 0.7276280751687296,
    "colsample_bytree": 0.5543816813470782,
    "reg_lambda": 19.192849668958537,
    "reg_alpha": 10.047676285972248
  }
}
```

## Test Set Evaluation
```json
{
  "model_path": "results/models/lightgbm.joblib",
  "parquet_path": "data/dataset_splits_encoded.parquet",
  "target_column": "weight-(kg)",
  "metrics": {
    "mae": 0.7387944704798104,
    "mse": 1.3399886258899096,
    "rmse": 1.157578777401309,
    "r2": 0.997039670071143,
    "mape": 0.010348572525013129,
    "median_ae": 0.49981615653896583,
    "explained_variance": 0.9970453364805728,
    "y_std": 21.275535405643144,
    "pred_std": 20.859328166811448,
    "residual_std": 1.1564703777828835
  },
  "shap": {
    "plot_path": "plots/shape/LightGBM/lightgbm_shap_beeswarm.png",
    "expected_value": 73.79339574733287,
    "feature_impacts": [
      {
        "feature": "water-intake-(liters)",
        "max_positive": 15.229419915323632,
        "max_negative": -13.193814021899033,
        "positive_rate": 0.42225,
        "negative_rate": 0.57775,
        "max_intensity": 15.229419915323632
      },
      {
        "feature": "cholesterol-mg",
        "max_positive": 11.118093096810686,
        "max_negative": -5.813236885093438,
        "positive_rate": 0.48025,
        "negative_rate": 0.51975,
        "max_intensity": 11.118093096810686
      },
      {
        "feature": "age",
        "max_positive": 9.191505977124935,
        "max_negative": -5.5631808658071735,
        "positive_rate": 0.483,
        "negative_rate": 0.517,
        "max_intensity": 9.191505977124935
      }
    ]
  }
}
```

> ℹ️ Les listes complètes de `feature_impacts` et des métriques sont disponibles dans les JSON versionnés. Les extraits ci-dessus résument les valeurs essentielles tout en illustrant les chemins relatifs.
