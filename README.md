# ML-Powered YouTube Sentiment Analysis System

This repository implements an end-to-end DataOps, ModelOps, DevOps, and monitoring workflow to classify comments into negative, neutral, and positive sentiment.

## What Is Implemented

- Data ingestion from YouTube Data API v3
- Supplementary Reddit dataset ingestion from local CSV/parquet path
- Text normalization and sentiment target engineering
- Great Expectations dataset validation
- Parquet export and DVC tracking support
- Multi-model training with sklearn Pipelines
- MLflow run tracking and champion model registration
- FastAPI inference API with single and batch endpoints
- Streamlit frontend that calls API endpoints via HTTP
- Prediction logging and Evidently drift report generation
- Dockerfiles and docker-compose orchestration

## Core Files

- data_pipeline.py
- dataops_utils.py
- models.py
- fastapi_app.py
- streamlit_app.py
- monitoring/generate_drift_report.py
- app_config.py
- docker-compose.yml
- requirements.txt

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a local environment file from the template and fill values.

```bash
copy .env.example .env
```

Required values:

- YOUTUBE_VIDEOS_API_KEY
- YOUTUBE_COMMENTS_API_KEY
- YOUTUBE_STATS_API_KEY
- REDDIT_DATASET_PATH (optional but recommended)
- HAND_LABELED_TEST_PATH (required for champion promotion)

Optional gate controls:

- PROMOTION_METRIC (default: macro_f1)
- MIN_HAND_LABELED_MACRO_F1 (default: 0.65)
- MIN_NEGATIVE_RECALL (default: 0.55)
- MIN_CLASS_RATIO (default: 0.10)
- REQUIRE_PROBABILISTIC_CHAMPION (default: true)
- MAX_BATCH_SIZE (default: 1000)

## End-to-End Run Order

1. Build and validate dataset.

```bash
python data_pipeline.py
```

2. Track parquet with DVC.

```bash
dvc add sentiment_analysis_data.parquet
```

3. Train models, evaluate against holdout and hand-labeled sets, and register the champion model in MLflow.

```bash
python models.py
```

Promotion policy:

- Selection metric is hand-labeled macro F1
- Champion must pass minimum hand-labeled macro F1 and negative-recall thresholds
- Champion must be probabilistic (LightGBM or XGBoost) when `REQUIRE_PROBABILISTIC_CHAMPION=true`

The winner is registered as `YouTube_Sentiment_Champion` and assigned the `champion` alias for serving.

4. Start MLflow UI.

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

5. Start FastAPI backend.

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

6. Start Streamlit frontend.

```bash
streamlit run streamlit_app.py
```

7. Generate drift reports after inference logs are created.

```bash
python monitoring/generate_drift_report.py
```

Reports are generated in monitoring/reports.

## API Contract

- POST /predict
	- Body: {"comment": "text"}
	- Returns: prediction_class, prediction_label, confidence

- POST /predict_batch
	- Body: {"comments": ["text1", "text2"]}
	- Returns: results array with prediction_class, prediction_label, confidence
	- Enforces MAX_BATCH_SIZE guardrail

## Docker Compose

Run all services (MLflow, FastAPI, Streamlit):

```bash
docker-compose up --build
```

Exposed ports:

- MLflow UI: 5000
- FastAPI: 8000
- Streamlit: 8501
