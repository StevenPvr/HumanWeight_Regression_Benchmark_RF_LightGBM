# Random Forest Evaluation Results

## Source Artefacts
- `results/best_random_forest_params.json`
- `results/models/random_forest_metrics.json`
- `results/eval/random_forest_test_metrics.json`

## Hyperparameter Optimisation
```json
{
  "best_params": {
    "n_estimators": 293,
    "max_depth": 29,
    "min_samples_split": 11,
    "min_samples_leaf": 1,
    "max_features": 0.3000600576598142,
    "bootstrap": false
  },
  "best_value": 0.6414507181608485,
  "val_metrics": {
    "mse": 0.6414507181608485
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
  "model_path": "/Users/steven/Documents/Programmation/Weigh_LifeStyle/results/models/random_forest.joblib",
  "parquet_path": "/Users/steven/Documents/Programmation/Weigh_LifeStyle/data/dataset_splits_encoded.parquet",
  "params_path": "/Users/steven/Documents/Programmation/Weigh_LifeStyle/results/best_random_forest_params.json",
  "target_column": "weight-(kg)",
  "random_state": 123,
  "validation": {
    "mse": 0.6414507181608483
  },
  "training_rows": 11999,
  "validation_rows": 4001,
  "best_params": {
    "n_estimators": 293,
    "max_depth": 29,
    "min_samples_split": 11,
    "min_samples_leaf": 1,
    "max_features": 0.3000600576598142,
    "bootstrap": false
  }
}
```

## Test Set Evaluation
```json
{
  "model_path": "/Users/steven/Documents/Programmation/Weigh_LifeStyle/results/models/random_forest.joblib",
  "parquet_path": "/Users/steven/Documents/Programmation/Weigh_LifeStyle/data/dataset_splits_encoded.parquet",
  "target_column": "weight-(kg)",
  "metrics": {
    "mae": 0.518716607156941,
    "mse": 0.6681698018539312,
    "rmse": 0.8174165412162462,
    "r2": 0.9985238657823137,
    "mape": 0.007328054182188059,
    "median_ae": 0.3654760618124939,
    "explained_variance": 0.9985256733039579,
    "y_std": 21.275535405643144,
    "pred_std": 20.956183954624464,
    "residual_std": 0.8169159259442765
  },
  "shap": null
}
```
