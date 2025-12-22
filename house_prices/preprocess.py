from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "SalePrice"
ID_COL = "Id"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small safe fixes before preprocessing."""
    out = df.copy()
    if "MSSubClass" in out.columns:
        out["MSSubClass"] = out["MSSubClass"].astype(str)
    return out


def make_one_hot_encoder() -> OneHotEncoder:
    """Sklearn-version-safe OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def fit_preprocessors(x_train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit all preprocessing objects on train only (no leakage)."""
    x_train_df = prepare_features(x_train_df)
    x_no_id = x_train_df.drop(columns=[ID_COL], errors="ignore")

    cat_cols = x_no_id.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_no_id.columns if c not in cat_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    encoder = make_one_hot_encoder()

    # fit then transform (no fit_transform)
    train_num = num_imputer.fit(x_no_id[num_cols]).transform(x_no_id[num_cols])
    scaler.fit(train_num)

    train_cat = cat_imputer.fit(x_no_id[cat_cols]).transform(x_no_id[cat_cols])
    encoder.fit(train_cat)

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "scaler": scaler,
        "encoder": encoder,
    }


def transform_data(x_df: pd.DataFrame, pre: dict[str, Any]) -> csr_matrix:
    """Transform data using fitted preprocessors."""
    x_df = prepare_features(x_df)
    x_no_id = x_df.drop(columns=[ID_COL], errors="ignore")

    num_cols = pre["num_cols"]
    cat_cols = pre["cat_cols"]

    num_imputer = pre["num_imputer"]
    cat_imputer = pre["cat_imputer"]
    scaler = pre["scaler"]
    encoder = pre["encoder"]
    expected = list(num_cols) + list(cat_cols)
    for c in expected:
        if c not in x_no_id.columns:
            x_no_id[c] = np.nan
    x_no_id = x_no_id[expected]
    num_arr = num_imputer.transform(x_no_id[num_cols])
    num_scaled = scaler.transform(num_arr)

    cat_arr = cat_imputer.transform(x_no_id[cat_cols])
    cat_encoded = encoder.transform(cat_arr)

    return hstack([csr_matrix(num_scaled), cat_encoded]).tocsr()


def rmsle_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSLE equals RMSE on log1p(target)."""
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))
