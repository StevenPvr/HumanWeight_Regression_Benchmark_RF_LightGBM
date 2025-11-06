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
    "n_estimators": 1240,
    "learning_rate": 0.033475989928947014,
    "num_leaves": 317,
    "max_depth": 14,
    "min_child_samples": 29,
    "subsample": 0.9857826959874048,
    "colsample_bytree": 0.5002531173970539,
    "reg_lambda": 73.91530744014005,
    "reg_alpha": 17.590104982168974
  },
  "best_value": 2.036410852782807,
  "data_summary": {
    "train_rows": 16000
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
  "training_rows": 16000,
  "best_params": {
    "n_estimators": 1166,
    "learning_rate": 0.08495160621047627,
    "num_leaves": 257,
    "max_depth": 18,
    "min_child_samples": 100,
    "subsample": 0.5613277034603019,
    "colsample_bytree": 0.86405352262954,
    "reg_lambda": 69.21219239466598,
    "reg_alpha": 11.188984942065268
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
    "mae": 0.8192848918781521,
    "mse": 1.489811677015039,
    "rmse": 1.2205784190354338,
    "r2": 0.9967086779614274,
    "mape": 0.011529640191814468,
    "median_ae": 0.5806108939152779,
    "explained_variance": 0.9967222180594248,
    "y_std": 21.275535405643144,
    "pred_std": 20.90356165958078,
    "residual_std": 1.2180651761004289
  },
  "shap": {
    "plot_path": "plots/shape/LightGBM/lightgbm_shap_beeswarm.png",
    "expected_value": 73.78933986822379,
    "feature_impacts": [
      {
        "feature": "cholesterol-mg",
        "max_positive": 13.343241149667108,
        "max_negative": -6.937739908127126,
        "positive_rate": 0.454,
        "negative_rate": 0.546,
        "max_intensity": 13.343241149667108
      },
      {
        "feature": "session-duration-(hours)",
        "max_positive": 11.813747535691874,
        "max_negative": -7.502113948098886,
        "positive_rate": 0.4815,
        "negative_rate": 0.5185,
        "max_intensity": 11.813747535691874
      },
      {
        "feature": "max-bpm",
        "max_positive": 9.764742009154322,
        "max_negative": -5.791669320944521,
        "positive_rate": 0.45325,
        "negative_rate": 0.54675,
        "max_intensity": 9.764742009154322
      }
    ]
  }
}
```

> ℹ️ Les listes complètes de `feature_impacts` et des métriques sont disponibles dans les JSON versionnés. Les extraits ci-dessus résument les valeurs essentielles tout en illustrant les chemins relatifs.
