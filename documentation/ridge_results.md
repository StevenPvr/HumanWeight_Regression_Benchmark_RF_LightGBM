# Ridge Regression Evaluation Results

## Source Artefacts
- `results/best_ridge_params.json`
- `results/models/ridge.joblib`
- `results/eval/ridge_test_metrics.json`

## Hyperparameter Optimisation
```json
{
  "best_params": {
    "alpha": 6.348755301423526
  },
  "best_value": 441.5643318526264,
  "cv_summary": {
    "cv_mse_mean": 441.5643318526264,
    "cv_rmse_mean": 21.009974381983287,
    "cv_mae_mean": 16.76103558342554,
    "cv_r2_mean": 0.012373697662376459,
    "cv_mse": 441.5643318526264
  }
}
```

## Test Set Evaluation
```json
{
  "model_path": "results/models/ridge.joblib",
  "parquet_path": "data/dataset_splits_encoded.parquet",
  "target_column": "weight-(kg)",
  "metrics": {
    "mae": 16.811451805430288,
    "mse": 444.8367566457085,
    "rmse": 21.091153516242503,
    "r2": 0.01725765524360645,
    "mape": 0.23784660528437052,
    "median_ae": 14.02331695748775,
    "explained_variance": 0.017841893552912658,
    "y_std": 21.275535405643144,
    "pred_std": 2.829443297315247,
    "residual_std": 21.084883260426448
  },
  "shap": null
}
```
