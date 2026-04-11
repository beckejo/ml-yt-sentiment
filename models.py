import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import LinearSVC

# ---------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------

#df = pd.read_csv("YoutubeCommentsDataSet.csv")  # rename if needed
#df.columns = ["comment", "sentiment"]  # ensure correct names
#df = pd.read_parquet('sentiment_analysis_data_20260410.parquet')
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
# 3. TF‑IDF vectorization
# ---------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

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

xgb_model.fit(X_train_tfidf, y_train)
lgb_model.fit(X_train_tfidf, y_train)
svc_model.fit(X_train_tfidf, y_train)


# ---------------------------------------------------
# 5. Predictions + evaluation
# ---------------------------------------------------
xgb_y_pred = xgb_model.predict(X_test_tfidf)
lgb_y_pred = lgb_model.predict(X_test_tfidf)
svc_y_pred = svc_model.predict(X_test_tfidf)

result_df = pd.DataFrame({
    'model': ['XGB', 'LGB', 'SVC'],
    'f1_score': [
        f1_score(y_test, xgb_y_pred, average = 'weighted'),
        f1_score(y_test, lgb_y_pred, average = 'weighted'),
        f1_score(y_test, svc_y_pred, average = 'weighted')
    ]
}).sort_values(by = 'f1_score', ascending = False)

result_df

selected_model = result_df.iloc[0]

selected_model