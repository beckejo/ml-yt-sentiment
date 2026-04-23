from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app_config import get_settings
from data_pipeline import clean_text


CLASS_LABELS = [0, 1, 2]
PROMOTABLE_MODELS = {"XGBoost", "LightGBM"}


def configure_mlflow_tracking() -> str:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.set_experiment("Default")
    return settings.mlflow_tracking_uri


def _normalize_sentiment(raw_value: object) -> int | None:
    if pd.isna(raw_value):
        return None

    if isinstance(raw_value, (int, float)):
        label = int(raw_value)
        if label in (0, 1, 2):
            return label
        return None

    lookup = {
        "negative": 0,
        "neg": 0,
        "neutral": 1,
        "neu": 1,
        "positive": 2,
        "pos": 2,
    }
    return lookup.get(str(raw_value).strip().lower())


def _load_text_classification_frame(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    comment_column_candidates = ["comment", "clean_comment", "text", "body"]
    sentiment_column_candidates = ["sentiment", "label", "target"]

    comment_col = next((column for column in comment_column_candidates if column in frame.columns), None)
    sentiment_col = next((column for column in sentiment_column_candidates if column in frame.columns), None)

    if comment_col is None or sentiment_col is None:
        raise ValueError(
            "Dataset must include a comment column (comment/clean_comment/text/body) and sentiment column (sentiment/label/target)."
        )

    output_frame = pd.DataFrame()
    output_frame["clean_comment"] = frame[comment_col].astype(str).apply(clean_text)
    output_frame["sentiment"] = frame[sentiment_col].apply(_normalize_sentiment)
    output_frame = output_frame.dropna(subset=["clean_comment", "sentiment"])
    output_frame["sentiment"] = output_frame["sentiment"].astype("int64")
    return output_frame


def _load_hand_labeled_evaluation_frame(dataset_path: str) -> pd.DataFrame:
    if not dataset_path:
        return pd.DataFrame(columns=["clean_comment", "sentiment"])

    return _load_text_classification_frame(dataset_path)


def _class_distribution(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts().reindex(CLASS_LABELS, fill_value=0)
    total = int(counts.sum())
    max_count = int(counts.max()) if total > 0 else 0
    min_count = int(counts.min()) if total > 0 else 0
    ratio = float(min_count / max_count) if max_count > 0 else 0.0

    payload: dict[str, float] = {
        "rows": float(total),
        "class_ratio_min_to_max": ratio,
    }
    for class_id in CLASS_LABELS:
        class_count = int(counts.loc[class_id])
        payload[f"class_{class_id}_count"] = float(class_count)
        payload[f"class_{class_id}_pct"] = float((class_count / total) * 100.0 if total > 0 else 0.0)
    return payload


def _compute_metrics(y_true: pd.Series, y_pred: Any, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{prefix}_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    for idx, class_id in enumerate(CLASS_LABELS):
        metrics[f"{prefix}_class_{class_id}_precision"] = float(class_precision[idx])
        metrics[f"{prefix}_class_{class_id}_recall"] = float(class_recall[idx])
        metrics[f"{prefix}_class_{class_id}_f1"] = float(class_f1[idx])
        metrics[f"{prefix}_class_{class_id}_support"] = float(class_support[idx])

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()
    metrics[f"{prefix}_confusion_matrix_json"] = float(0.0)  # placeholder for uniform payload type
    metrics[f"{prefix}_negative_recall"] = metrics[f"{prefix}_class_0_recall"]
    metrics[f"{prefix}_positive_recall"] = metrics[f"{prefix}_class_2_recall"]
    metrics[f"{prefix}_neutral_recall"] = metrics[f"{prefix}_class_1_recall"]
    metrics[f"{prefix}_confusion_matrix"] = matrix  # type: ignore[assignment]
    return metrics


def _metric_key_for_promotion(metric_name: str) -> str:
    mapping = {
        "macro_f1": "hand_labeled_macro_f1",
        "weighted_f1": "hand_labeled_f1_weighted",
    }
    if metric_name not in mapping:
        raise ValueError(f"Unsupported PROMOTION_METRIC={metric_name}. Supported values: {sorted(mapping.keys())}")
    return mapping[metric_name]


def train_and_track_models() -> pd.DataFrame:
    settings = get_settings()
    tracking_uri = configure_mlflow_tracking()
    print(f"MLflow tracking URI: {tracking_uri}")

    if not settings.hand_labeled_test_path:
        raise ValueError("HAND_LABELED_TEST_PATH is required for champion promotion.")

    raw_df = pd.read_parquet(settings.data_output_path)
    required_columns = ["clean_comment", "sentiment"]
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Training dataset is missing required columns: {missing_columns}")

    selected_columns = required_columns + (["source"] if "source" in raw_df.columns else [])
    df = raw_df[selected_columns].dropna(subset=["clean_comment", "sentiment"]).copy()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype("int64")
    if "source" not in df.columns:
        df["source"] = "unknown"

    reddit_df = df[df["source"].astype(str).str.lower() == "reddit"].copy()
    if not reddit_df.empty and reddit_df["sentiment"].nunique() >= 3:
        training_df = reddit_df.copy()
        training_source = "reddit"
    else:
        training_df = df.copy()
        training_source = "mixed"
    hand_labeled_eval_df = _load_hand_labeled_evaluation_frame(settings.hand_labeled_test_path)

    if hand_labeled_eval_df.empty:
        raise ValueError("HAND_LABELED_TEST_PATH is set but produced an empty evaluation dataset.")
    if hand_labeled_eval_df["sentiment"].nunique() < 3:
        raise ValueError("Hand-labeled evaluation set must contain all three sentiment classes.")

    if training_df["sentiment"].nunique() < 2:
        raise ValueError("Training data must contain at least two sentiment classes after source filtering.")

    initial_distribution = _class_distribution(training_df["sentiment"])
    if initial_distribution["class_ratio_min_to_max"] < settings.min_class_ratio and df["sentiment"].nunique() >= 3:
        training_df = df.copy()
        training_source = "mixed_imbalance_fallback"

    training_distribution = _class_distribution(training_df["sentiment"])
    hand_labeled_distribution = _class_distribution(hand_labeled_eval_df["sentiment"])
    print(
        "Training distribution ratio min/max:",
        f"{training_distribution['class_ratio_min_to_max']:.4f}",
        "| Hand-labeled distribution ratio min/max:",
        f"{hand_labeled_distribution['class_ratio_min_to_max']:.4f}",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        training_df["clean_comment"],
        training_df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=training_df["sentiment"],
    )

    train_split_distribution = _class_distribution(y_train)
    holdout_distribution = _class_distribution(y_test)

    tfidf_params = {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "stop_words": "english",
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.1,
        "max_depth": -1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
    }
    svc_params = {
        "class_weight": "balanced",
        "random_state": 42,
    }

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model_specs = [
        (
            "XGBoost",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(**tfidf_params)),
                    ("classifier", XGBClassifier(**xgb_params)),
                ]
            ),
            "xgboost_pipeline",
            xgb_params,
            {"classifier__sample_weight": sample_weights},
        ),
        (
            "LightGBM",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(**tfidf_params)),
                    ("classifier", lgb.LGBMClassifier(**lgb_params)),
                ]
            ),
            "lightgbm_pipeline",
            lgb_params,
            {"classifier__sample_weight": sample_weights},
        ),
        (
            "LinearSVC",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(**tfidf_params)),
                    ("classifier", LinearSVC(**svc_params)),
                ]
            ),
            "linearsvc_pipeline",
            svc_params,
            {"classifier__sample_weight": sample_weights},
        ),
    ]

    promotion_metric_key = _metric_key_for_promotion(settings.promotion_metric)

    run_results: list[dict[str, str | float]] = []

    for model_name, pipeline, artifact_path, model_params, fit_kwargs in model_specs:
        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(X_train, y_train, **fit_kwargs)
            y_pred = pipeline.predict(X_test)
            holdout_metrics = _compute_metrics(y_test, y_pred, prefix="holdout")

            hand_labeled_pred = pipeline.predict(hand_labeled_eval_df["clean_comment"])
            hand_labeled_metrics = _compute_metrics(
                hand_labeled_eval_df["sentiment"],
                hand_labeled_pred,
                prefix="hand_labeled",
            )

            selection_metric = float(hand_labeled_metrics[promotion_metric_key])

            promotable_model = (not settings.require_probabilistic_champion) or (model_name in PROMOTABLE_MODELS)
            threshold_ok = (
                hand_labeled_metrics["hand_labeled_macro_f1"] >= settings.min_hand_labeled_macro_f1
                and hand_labeled_metrics["hand_labeled_negative_recall"] >= settings.min_negative_recall
            )
            promotion_eligible = promotable_model and threshold_ok
            ineligibility_reason = ""
            if not promotable_model:
                ineligibility_reason = "non_promotable_model_type"
            elif not threshold_ok:
                ineligibility_reason = "quality_threshold_not_met"

            mlflow.log_params({f"tfidf_{k}": v for k, v in tfidf_params.items()})
            mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
            mlflow.log_param("imbalance_strategy", "balanced_sample_weight")
            mlflow.log_param("training_source", training_source)
            mlflow.log_param("promotion_metric", settings.promotion_metric)
            mlflow.log_param("promotion_requires_probabilistic", settings.require_probabilistic_champion)
            mlflow.log_param("promotion_model_promotable", promotable_model)
            mlflow.log_param("promotion_eligible", promotion_eligible)
            mlflow.log_param("promotion_ineligibility_reason", ineligibility_reason)
            mlflow.log_param("promotion_threshold_macro_f1", settings.min_hand_labeled_macro_f1)
            mlflow.log_param("promotion_threshold_negative_recall", settings.min_negative_recall)
            mlflow.log_param("promotion_min_class_ratio", settings.min_class_ratio)
            mlflow.log_param("training_rows", int(len(training_df)))
            mlflow.log_param("holdout_rows", int(len(X_test)))
            mlflow.log_param("hand_labeled_eval_rows", int(len(hand_labeled_eval_df)))

            for metric_name, metric_value in training_distribution.items():
                mlflow.log_param(f"training_distribution_{metric_name}", metric_value)
            for metric_name, metric_value in train_split_distribution.items():
                mlflow.log_param(f"train_split_distribution_{metric_name}", metric_value)
            for metric_name, metric_value in holdout_distribution.items():
                mlflow.log_param(f"holdout_distribution_{metric_name}", metric_value)
            for metric_name, metric_value in hand_labeled_distribution.items():
                mlflow.log_param(f"hand_labeled_distribution_{metric_name}", metric_value)

            for metric_name, metric_value in holdout_metrics.items():
                if not metric_name.endswith("_confusion_matrix") and not metric_name.endswith("_confusion_matrix_json"):
                    mlflow.log_metric(metric_name, float(metric_value))
            for metric_name, metric_value in hand_labeled_metrics.items():
                if not metric_name.endswith("_confusion_matrix") and not metric_name.endswith("_confusion_matrix_json"):
                    mlflow.log_metric(metric_name, float(metric_value))

            mlflow.log_text(
                json.dumps({"labels": CLASS_LABELS, "matrix": holdout_metrics["holdout_confusion_matrix"]}, indent=2),
                "metrics/holdout_confusion_matrix.json",
            )
            mlflow.log_text(
                json.dumps({"labels": CLASS_LABELS, "matrix": hand_labeled_metrics["hand_labeled_confusion_matrix"]}, indent=2),
                "metrics/hand_labeled_confusion_matrix.json",
            )
            mlflow.log_metric("selection_metric", selection_metric)
            mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)

            run_results.append(
                {
                    "model": model_name,
                    "selection_score": selection_metric,
                    "holdout_macro_f1": holdout_metrics["holdout_macro_f1"],
                    "holdout_f1_weighted": holdout_metrics["holdout_f1_weighted"],
                    "hand_labeled_macro_f1": hand_labeled_metrics["hand_labeled_macro_f1"],
                    "hand_labeled_f1_weighted": hand_labeled_metrics["hand_labeled_f1_weighted"],
                    "hand_labeled_negative_recall": hand_labeled_metrics["hand_labeled_negative_recall"],
                    "promotion_eligible": float(1.0 if promotion_eligible else 0.0),
                    "ineligibility_reason": ineligibility_reason,
                    "run_id": run.info.run_id,
                    "artifact_path": artifact_path,
                }
            )

    result_df = pd.DataFrame(run_results).sort_values(by="selection_score", ascending=False)
    print(result_df)

    eligible_df = result_df[result_df["promotion_eligible"] == 1.0].copy()
    if eligible_df.empty:
        reasons = result_df[["model", "ineligibility_reason", "selection_score"]].to_dict(orient="records")
        raise ValueError(
            "No models passed promotion gates. Review hand-labeled metrics and thresholds. "
            f"Candidate status: {reasons}"
        )

    selected_model = eligible_df.iloc[0]
    champion_model_uri = f"runs:/{selected_model['run_id']}/{selected_model['artifact_path']}"
    registered_model = mlflow.register_model(model_uri=champion_model_uri, name="YouTube_Sentiment_Champion")
    MlflowClient().set_registered_model_alias(
        name="YouTube_Sentiment_Champion",
        alias="champion",
        version=registered_model.version,
    )

    summary = {
        "winner_model": selected_model["model"],
        "selection_metric": settings.promotion_metric,
        "selection_score": float(selected_model["selection_score"]),
        "holdout_macro_f1": float(selected_model["holdout_macro_f1"]),
        "holdout_f1_weighted": float(selected_model["holdout_f1_weighted"]),
        "hand_labeled_macro_f1": float(selected_model["hand_labeled_macro_f1"]),
        "hand_labeled_f1_weighted": float(selected_model["hand_labeled_f1_weighted"]),
        "hand_labeled_negative_recall": float(selected_model["hand_labeled_negative_recall"]),
        "min_hand_labeled_macro_f1": float(settings.min_hand_labeled_macro_f1),
        "min_negative_recall": float(settings.min_negative_recall),
        "require_probabilistic_champion": bool(settings.require_probabilistic_champion),
        "run_id": selected_model["run_id"],
        "model_uri": champion_model_uri,
        "registry_name": "YouTube_Sentiment_Champion",
        "registry_alias": "champion",
        "registry_version": str(registered_model.version),
    }

    with open("champion_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return result_df


if __name__ == "__main__":
    train_and_track_models()
