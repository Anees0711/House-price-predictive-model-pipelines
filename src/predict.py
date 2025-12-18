import os
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

from hp_preprocessing import fit_preprocessors, transform_features


def main(data_dir: str = "data", out_dir: str = "outputs", alpha: float = 20.0, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    TARGET = "SalePrice"
    ID_COL = "Id"

    X_train_raw = train_df.drop(columns=[TARGET]).copy()
    y_train_log = np.log1p(train_df[TARGET].copy())

    pre = fit_preprocessors(X_train_raw, id_col=ID_COL)
    X_train = transform_features(X_train_raw, pre)
    X_test  = transform_features(test_df.copy(), pre)

    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_train, y_train_log)

    pred_test = np.expm1(model.predict(X_test))

    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: pred_test})
    out_path = os.path.join(out_dir, "submission.csv")
    submission.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(submission.head(10))  # <= 20 rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(data_dir=args.data_dir, out_dir=args.out_dir, alpha=args.alpha, seed=args.seed)
