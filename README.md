# ML-Powered YouTube Sentiment Analysis System

An end-to-end machine learning pipeline that classifies YouTube comments as **positive**, **neutral**, or **negative**.

## Overview

As YouTube channels grow, it becomes impossible to manually review thousands of comments to understand audience feedback. Viral videos often attract noise, self-promotion, and trolling, which can bury meaningful criticism and sentiment.

This project addresses that problem with a machine learning system that processes unstructured text data using TF-IDF vectorization with bigrams and evaluates multiple models, including LightGBM, XGBoost, and LinearSVC.

The system includes:

- Data ingestion through the YouTube Data API v3
- Data validation with Great Expectations
- Data versioning with Data Version Control (DVC)
- Experiment tracking and model registration with MLflow

## Team Members

**Group 7**

- Precious Adugyamfi
- Shelby Altman-Metzler
- Jay Becker
- Julie Cella
- Christian Dinevski

## How to Run

Follow these steps to run the project locally.

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install pandas scikit-learn xgboost lightgbm dvc great_expectations mlflow fastapi streamlit uvicorn
```

### 2. Run the Data Pipeline

Execute the ingestion and validation pipeline:

```bash
python data_pipeline.py
```

Note: You must have active YouTube Data API v3 keys configured in the script to access the `commentsThreads` and `videos.list` endpoints.

This step pulls video statistics and comments, validates the schema, checks for null values, and verifies data types using Great Expectations. The cleaned dataset is saved as `sentiment_analysis_data.parquet`.

### 3. Track Data with DVC

Initialize DVC and track the dataset snapshot for reproducibility:

```bash
dvc init
dvc add sentiment_analysis_data.parquet
```

Be sure to commit the generated `.dvc` file to Git.

### 4. Train Models and Track Experiments

Run the modeling script to train the baseline models and identify the champion model:

```bash
python models.py
```

### 5. View the MLflow Dashboard

Start the MLflow UI to compare model runs, review F1 scores, and access the registered model:

```bash
mlflow ui
```

## Model Comparison

The project compares:

- XGBoost
- LightGBM
- LinearSVC

## DataOps and ModelOps Flow

1. Ingest YouTube comments and video metadata.
2. Validate the dataset with Great Expectations.
3. Version the cleaned data with DVC.
4. Train and evaluate candidate models.
5. Track metrics and register the best-performing model in MLflow.