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

## Project Structure
The repository is organized as follows to support a full DataOps and ModelOps workflow:
ml-yt-sentiment/
│── data_pipeline.py # Data ingestion and validation pipeline
│── sentiment_analysis_data.parquet # Processed dataset (DVC tracked)
│── modeling.py # Model training and evaluation
│── fastapi_app.py # API for serving predictions
│── streamlit_app.py # Frontend UI for interaction
│── mlflow.db # MLflow tracking database
│── README.md # Project documentation
│── requirements.txt # Python dependencies

This structure separates data ingestion, model training, and deployment components, making the system modular and easier to maintain.

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
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```

Run this command from the project root so the relative `sqlite:///mlflow.db` path resolves to this repository's tracking database.

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
