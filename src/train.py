import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge

from hp_preprocessing import fit_preprocessors, transform_features, rmsle_from_log


def main(data_dir: str = "data", alpha: float = 20.0, seed: int = 42):
    train_path = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_path)

    TARGET = "SalePrice"
    ID_COL = "Id"

    X_raw = train_df.drop(columns=[TARGET]).copy()
    y_log = np.log1p(train_df[TARGET].copy())

    # Holdout split (no leakage: fit preprocessors only on train split)
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_raw, y_log, test_size=0.2, random_state=seed
    )

    pre = fit_preprocessors(X_tr_raw, id_col=ID_COL)
    X_tr = transform_features(X_tr_raw, pre)
    X_va = transform_features(X_va_raw, pre)

    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)

    print(f"Holdout RMSLE: {rmsle_from_log(y_va, pred_va):.5f}")
    print(f"Numeric features: {len(pre['num_cols'])} | Categorical features: {len(pre['cat_cols'])}")

    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmsles = []

    X_all = X_raw.reset_index(drop=True)
    y_all = y_log.reset_index(drop=True)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
        X_tr_raw = X_all.iloc[tr_idx].copy()
        X_va_raw = X_all.iloc[va_idx].copy()
        y_tr = y_all.iloc[tr_idx].copy()
        y_va = y_all.iloc[va_idx].copy()

        pre = fit_preprocessors(X_tr_raw, id_col=ID_COL)
        X_tr = transform_features(X_tr_raw, pre)
        X_va = transform_features(X_va_raw, pre)

        m = Ridge(alpha=alpha, random_state=seed)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)

        fold_rmsle = rmsle_from_log(y_va, pred)
        rmsles.append(fold_rmsle)
        print(f"Fold {fold} RMSLE: {fold_rmsle:.5f}")

    print(f"CV RMSLE (mean ± std): {np.mean(rmsles):.5f} ± {np.std(rmsles):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(data_dir=args.data_dir, alpha=args.alpha, seed=args.seed)