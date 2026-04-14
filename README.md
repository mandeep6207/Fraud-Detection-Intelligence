# Fraud Detection Product

Production-style fraud detection system with a single orchestrated ML pipeline, persisted artifacts, FastAPI inference service, and API-driven Streamlit dashboard.

## System Flow

1. Load labeled transaction data from CSV.
2. Build preprocessing pipeline (imputation + one-hot encoding + scaling).
3. Train Logistic Regression and Random Forest.
4. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
5. Tune decision threshold using PR-curve for best F1.
6. Auto-select best model by configured metric (`f1` or `roc_auc`).
7. Save artifacts to `models/`:
	 - `model.pkl`
	 - `preprocessing_pipeline.pkl`
	 - `metrics.json`
8. Serve online predictions via FastAPI `/predict`.
9. Use Streamlit dashboard as an API client.

## Structure

```
src/
	data/
	features/
	models/
	evaluation/
	inference/
	api/
	pipeline/
	main.py
models/
	model.pkl
	preprocessing_pipeline.pkl
	metrics.json
dashboard/
	app.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.main train --data-path data/processed/fraud.csv --target-column is_fraud --selection-metric f1
```

## Evaluate Saved Model

```bash
python -m src.main evaluate --data-path data/processed/fraud.csv --target-column is_fraud
```

## Predict One Transaction

```bash
python -m src.main predict --input-json "{\"amount\": 1250, \"merchant\": \"StoreA\", \"transaction_type\": \"online\"}"
```

## Explain One Prediction

```bash
python -m src.main explain --input-json "{\"amount\": 1250, \"merchant\": \"StoreA\", \"transaction_type\": \"online\"}" --top-n 10
```

## Run API

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## API Contract

`POST /predict`

Request body:

```json
{
	"transaction": {
		"amount": 1250,
		"merchant": "StoreA",
		"transaction_type": "online"
	}
}
```

Response:

```json
{
	"fraud_probability": 0.82,
	"label": 1,
	"threshold": 0.47,
	"model": "random_forest"
}
```
