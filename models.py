import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import LinearSVC

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
# MLflow experiment setup
# ---------------------------------------------------
mlflow.set_experiment("youtube_sentiment_model_comparison")

# ---------------------------------------------------
# 4. Fit models
# ---------------------------------------------------
xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    learning_rate=0.1,
    max_depth=-1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

svc_model = LinearSVC()

# XGB
with mlflow.start_run(run_name="XGB"):
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    xgb_model.fit(X_train_tfidf, y_train)
    xgb_y_pred = xgb_model.predict(X_test_tfidf)
    xgb_f1 = f1_score(y_test, xgb_y_pred, average='weighted')
    mlflow.log_metric("f1_score", xgb_f1)
    mlflow.xgboost.log_model(xgb_model, artifact_path="model")

# LGB
with mlflow.start_run(run_name="LGB"):
    mlflow.log_param("model", "LGBMClassifier")
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", -1)
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    lgb_model.fit(X_train_tfidf, y_train)
    lgb_y_pred = lgb_model.predict(X_test_tfidf)
    lgb_f1 = f1_score(y_test, lgb_y_pred, average='weighted')
    mlflow.log_metric("f1_score", lgb_f1)
    mlflow.lightgbm.log_model(lgb_model, artifact_path="model")

# SVC
with mlflow.start_run(run_name="SVC"):
    mlflow.log_param("model", "LinearSVC")

    svc_model.fit(X_train_tfidf, y_train)
    svc_y_pred = svc_model.predict(X_test_tfidf)
    svc_f1 = f1_score(y_test, svc_y_pred, average='weighted')
    mlflow.log_metric("f1_score", svc_f1)
    mlflow.sklearn.log_model(svc_model, artifact_path="model")

# ---------------------------------------------------
# 5. Predictions + evaluation
# ---------------------------------------------------
result_df = pd.DataFrame({
    'model': ['XGB', 'LGB', 'SVC'],
    'f1_score': [
        xgb_f1,
        lgb_f1,
        svc_f1
    ]
}).sort_values(by='f1_score', ascending=False)

print(result_df)

selected_model = result_df.iloc[0]

print("\nSelected model:")
print(selected_model)
