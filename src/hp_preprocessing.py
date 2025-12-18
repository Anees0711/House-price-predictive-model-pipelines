import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Small fixes to reduce mixed-type issues."""
    df = df.copy()
    # MSSubClass is numeric but acts like a categorical label
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df


def make_onehot_encoder():
    """Version-safe OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def fit_preprocessors(X_fit: pd.DataFrame, id_col: str = "Id") -> dict:
    """
    Fit imputers/scaler/encoder on X_fit only (avoid leakage).
    Returns fitted objects + column lists.
    """
    X_fit = cast_columns(X_fit)

    if id_col in X_fit.columns:
        X_fit = X_fit.drop(columns=[id_col])

    cat_cols = X_fit.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_fit.columns if c not in cat_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    ohe = make_onehot_encoder()

    # Fit numeric
    X_num = num_imputer.fit_transform(X_fit[num_cols])
    scaler.fit(X_num)

    # Fit categorical
    X_cat = cat_imputer.fit_transform(X_fit[cat_cols])
    ohe.fit(X_cat)

    return {
        "id_col": id_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "scaler": scaler,
        "ohe": ohe,
    }


def transform_features(X: pd.DataFrame, pre: dict):
    """Transform any dataframe using already-fitted preprocessors."""
    X = cast_columns(X)

    id_col = pre["id_col"]
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    num_cols = pre["num_cols"]
    cat_cols = pre["cat_cols"]

    # Numeric
    X_num = pre["num_imputer"].transform(X[num_cols])
    X_num = pre["scaler"].transform(X_num)

    # Categorical
    X_cat = pre["cat_imputer"].transform(X[cat_cols])
    X_cat_ohe = pre["ohe"].transform(X_cat)

    # Combine
    return hstack([csr_matrix(X_num), X_cat_ohe]).tocsr()


def rmsle_from_log(y_true_log, y_pred_log) -> float:
    """RMSLE equals RMSE on log1p(target)."""
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))