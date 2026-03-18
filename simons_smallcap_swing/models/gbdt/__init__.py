"""
models.gbdt — Gradient boosted decision tree training pipeline.

    train_lgbm.py       LightGBM training with temporal splits, OOF/OOS, promotion gates
    train_xgb.py        XGBoost training with temporal splits, OOF/OOS, promotion gates
    calibrate.py        Platt / isotonic / temperature scaling on OOF scores
    tune_hyperparams.py HPO with robustness penalties (random search / Bayesian)
"""
