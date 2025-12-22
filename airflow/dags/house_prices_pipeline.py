from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# ----- Make /opt/project importable -----
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from house_prices.preprocess import (  # noqa: E402
    TARGET_COL,
    ID_COL,
    fit_preprocessors,
    transform_data,
    rmsle_from_log,
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

DEFAULT_ARGS = {"owner": "airflow", "retries": 1}


def _assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}. "
            f"Check your docker volume mount: ./data -> {DATA_DIR}"
        )


def new_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def pg_hook() -> PostgresHook:
    return PostgresHook(postgres_conn_id="hp_postgres")


def load_raw_to_postgres(**context) -> None:
    run_id = new_run_id()
    context["ti"].xcom_push(key="run_id", value=run_id)

    _assert_file_exists(TRAIN_PATH)
    _assert_file_exists(TEST_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    hook = pg_hook()
    conn = hook.get_conn()
    cur = conn.cursor()

    for _, row in train_df.iterrows():
        rid = int(row[ID_COL])
        payload = json.dumps(row.where(pd.notna(row), None).to_dict())
        cur.execute(
            """
            INSERT INTO raw.housing_train (id, payload)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
            """,
            (rid, payload),
        )

    for _, row in test_df.iterrows():
        rid = int(row[ID_COL])
        payload = json.dumps(row.where(pd.notna(row), None).to_dict())
        cur.execute(
            """
            INSERT INTO raw.housing_test (id, payload)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
            """,
            (rid, payload),
        )

    conn.commit()
    cur.close()
    conn.close()


def data_quality_check(**context) -> None:
    run_id = context["ti"].xcom_pull(key="run_id")

    _assert_file_exists(TRAIN_PATH)
    _assert_file_exists(TEST_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    def miss_rate(df: pd.DataFrame) -> float:
        return float(df.isna().mean().mean())

    hook = pg_hook()

    hook.run(
        """
        INSERT INTO monitoring.data_quality_runs (run_id, dataset, missing_rate, row_count)
        VALUES (%s, 'train', %s, %s)
        ON CONFLICT (run_id, dataset) DO UPDATE SET
          missing_rate=EXCLUDED.missing_rate, row_count=EXCLUDED.row_count
        """,
        parameters=(run_id, miss_rate(train_df), int(len(train_df))),
    )

    hook.run(
        """
        INSERT INTO monitoring.data_quality_runs (run_id, dataset, missing_rate, row_count)
        VALUES (%s, 'test', %s, %s)
        ON CONFLICT (run_id, dataset) DO UPDATE SET
          missing_rate=EXCLUDED.missing_rate, row_count=EXCLUDED.row_count
        """,
        parameters=(run_id, miss_rate(test_df), int(len(test_df))),
    )


def train_and_evaluate(**context) -> None:
    run_id = context["ti"].xcom_pull(key="run_id")

    _assert_file_exists(TRAIN_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=[TARGET_COL]).copy()
    y = train_df[TARGET_COL].copy()

    # Split BEFORE preprocessing (avoid leakage)
    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_log = np.log1p(y_train_raw.to_numpy())
    y_test_log = np.log1p(y_test_raw.to_numpy())

    pre = fit_preprocessors(X_train_df)
    X_train_mat = transform_data(X_train_df, pre)
    X_test_mat = transform_data(X_test_df, pre)

    model = Ridge(alpha=20.0)
    model.fit(X_train_mat, y_train_log)

    pred_test_log = model.predict(X_test_mat)
    rmsle = rmsle_from_log(y_test_log, pred_test_log)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.joblib")
    preproc_path = os.path.join(MODELS_DIR, f"preprocessors_{run_id}.joblib")

    dump(model, model_path)
    dump(pre, preproc_path)

    hook = pg_hook()

    hook.run(
        """
        INSERT INTO model.model_runs (run_id, model_name, params, train_rows, test_rows, model_path, preproc_path)
        VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING
        """,
        parameters=(
            run_id,
            "Ridge",
            json.dumps({"alpha": 20.0}),
            int(len(X_train_df)),
            int(len(X_test_df)),
            model_path,
            preproc_path,
        ),
    )

    hook.run(
        """
        INSERT INTO model.model_metrics (run_id, metric_name, metric_value)
        VALUES (%s, %s, %s)
        ON CONFLICT (run_id, metric_name) DO UPDATE SET metric_value=EXCLUDED.metric_value
        """,
        parameters=(run_id, "rmsle", float(rmsle)),
    )


def batch_score_test(**context) -> None:
    run_id = context["ti"].xcom_pull(key="run_id")

    _assert_file_exists(TEST_PATH)

    test_df = pd.read_csv(TEST_PATH)

    model = load(os.path.join(MODELS_DIR, f"model_{run_id}.joblib"))
    pre = load(os.path.join(MODELS_DIR, f"preprocessors_{run_id}.joblib"))

    X_mat = transform_data(test_df, pre)
    pred_log = model.predict(X_mat)
    pred = np.expm1(pred_log)

    hook = pg_hook()
    conn = hook.get_conn()
    cur = conn.cursor()

    for idx, price in zip(test_df[ID_COL].astype(int).tolist(), pred.tolist()):
        cur.execute(
            """
            INSERT INTO model.predictions_batch (run_id, id, prediction)
            VALUES (%s, %s, %s)
            ON CONFLICT (run_id, id) DO UPDATE SET prediction=EXCLUDED.prediction
            """,
            (run_id, int(idx), float(price)),
        )

    conn.commit()
    cur.close()
    conn.close()


with DAG(
    dag_id="house_prices_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["house-prices", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="load_raw_to_postgres", python_callable=load_raw_to_postgres)
    t2 = PythonOperator(task_id="data_quality_check", python_callable=data_quality_check)
    t3 = PythonOperator(task_id="train_and_evaluate", python_callable=train_and_evaluate)
    t4 = PythonOperator(task_id="batch_score_test", python_callable=batch_score_test)

    t1 >> t2 >> t3 >> t4
