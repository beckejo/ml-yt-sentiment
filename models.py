import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import LinearSVC
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn


def configure_mlflow_tracking() -> str:
    # Build an absolute sqlite URI from this script location so shell state cannot override targets.
    project_root = Path(__file__).resolve().parent
    db_path = project_root / "mlflow.db"
    # Use absolute path with forward slashes for Windows compatibility
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("Default")
    return tracking_uri


TRACKING_URI = configure_mlflow_tracking()
print(f"MLflow tracking URI: {TRACKING_URI}")

# ---------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------

# df = pd.read_csv("YoutubeCommentsDataSet.csv")  # rename if needed
# df.columns = ["comment", "sentiment"]  # ensure correct names
# df = pd.read_parquet('sentiment_analysis_data_20260410.parquet')
df = pd.read_parquet('sentiment_analysis_data.parquet')
df = df[['video_id', 'clean_comment', 'sentiment']]

# ---------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_comment"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# ---------------------------------------------------
# 3. TF-IDF vectorization
# ---------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------------------------------------------
# 4. Fit, evaluate, and track models with MLflow
# ---------------------------------------------------
run_results = []

with mlflow.start_run(run_name="XGBoost") as run:
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_tfidf, y_train)
    xgb_y_pred = xgb_model.predict(X_test_tfidf)
    xgb_f1 = f1_score(y_test, xgb_y_pred, average='weighted')

    mlflow.log_params(xgb_params)
    mlflow.log_metric("f1_score_weighted", xgb_f1)
    mlflow.xgboost.log_model(xgb_model, artifact_path="xgboost_model")

    run_results.append({
        "model": "XGB",
        "f1_score": xgb_f1,
        "run_id": run.info.run_id,
        "artifact_path": "xgboost_model"
    })

with mlflow.start_run(run_name="LightGBM") as run:
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.1,
        "max_depth": -1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_tfidf, y_train)
    lgb_y_pred = lgb_model.predict(X_test_tfidf)
    lgb_f1 = f1_score(y_test, lgb_y_pred, average='weighted')

    mlflow.log_params(lgb_params)
    mlflow.log_metric("f1_score_weighted", lgb_f1)
    mlflow.lightgbm.log_model(lgb_model, artifact_path="lightgbm_model")

    run_results.append({
        "model": "LGB",
        "f1_score": lgb_f1,
        "run_id": run.info.run_id,
        "artifact_path": "lightgbm_model"
    })

with mlflow.start_run(run_name="LinearSVC") as run:
    svc_model = LinearSVC()
    svc_params = svc_model.get_params()
    svc_model.fit(X_train_tfidf, y_train)
    svc_y_pred = svc_model.predict(X_test_tfidf)
    svc_f1 = f1_score(y_test, svc_y_pred, average='weighted')

    mlflow.log_params(svc_params)
    mlflow.log_metric("f1_score_weighted", svc_f1)
    mlflow.sklearn.log_model(svc_model, artifact_path="linearsvc_model")

    run_results.append({
        "model": "SVC",
        "f1_score": svc_f1,
        "run_id": run.info.run_id,
        "artifact_path": "linearsvc_model"
    })

result_df = pd.DataFrame(run_results).sort_values(by='f1_score', ascending=False)

print(result_df)

selected_model = result_df.iloc[0]

champion_model_uri = f"runs:/{selected_model['run_id']}/{selected_model['artifact_path']}"
mlflow.register_model(
    model_uri=champion_model_uri,
    name="YouTube_Sentiment_Champion"
)

selected_model
