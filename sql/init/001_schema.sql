CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS model;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS raw.housing_train (
  id INT PRIMARY KEY,
  payload JSONB NOT NULL,
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.housing_test (
  id INT PRIMARY KEY,
  payload JSONB NOT NULL,
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model.model_runs (
  run_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  params JSONB NOT NULL,
  train_rows INT NOT NULL,
  test_rows INT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  model_path TEXT NOT NULL,
  preproc_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model.model_metrics (
  run_id TEXT NOT NULL REFERENCES model.model_runs(run_id) ON DELETE CASCADE,
  metric_name TEXT NOT NULL,
  metric_value DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS model.predictions_batch (
  run_id TEXT NOT NULL,
  id INT NOT NULL,
  prediction DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (run_id, id)
);

CREATE TABLE IF NOT EXISTS monitoring.data_quality_runs (
  run_id TEXT NOT NULL,
  dataset TEXT NOT NULL CHECK (dataset IN ('train','test')),
  missing_rate DOUBLE PRECISION NOT NULL,
  row_count INT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (run_id, dataset)
);

CREATE TABLE IF NOT EXISTS app.inference_requests (
  request_id BIGSERIAL PRIMARY KEY,
  request_payload JSONB NOT NULL,
  predicted_price DOUBLE PRECISION NOT NULL,
  model_run_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
