# House-price-predictive model & pipelines

End-to-end House Price prediction using the Ames Housing (Kaggle House Prices) dataset.

What’s included:
- Load train/test data
- Handle missing values
- Scale numeric (continuous) features
- One-hot encode categorical features
- Train Ridge regression
- Evaluate with RMSLE (holdout + K-Fold)
- Generate `outputs/submission.csv`

## Folder structure
- `data/` : train/test CSVs
- `notebooks/` : main notebook (keep outputs for GitHub grading)
- `src/` : reusable code (no Pipeline / no ColumnTransformer)
- `outputs/` : generated artifacts

## Run (Notebook)
Open `notebooks/house-prices-modeling.ipynb`, Run All, Save (do not clear outputs).

## Run (Scripts)
```bash
pip install -r requirements.txt

python src/train.py --data_dir data
python src/predict.py --data_dir data --out_dir outputs
