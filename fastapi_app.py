from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app_config import get_settings


SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=1)


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    confidence: float | None


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


settings = get_settings()
app = FastAPI(title="YouTube Sentiment Inference API")
model: Any | None = None
model_metadata: dict[str, Any] = {}


def _predict_single_text(comment_text: str) -> tuple[int, float | None]:
    """Predict one comment with input-shape fallbacks for sklearn and pyfunc models."""
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Try text-native inputs first (for sklearn pipelines), then DataFrame (for pyfunc).
    candidate_inputs: list[Any] = [
        pd.Series([comment_text], name="clean_comment"),
        [comment_text],
        pd.DataFrame({"clean_comment": [comment_text]}),
    ]

    prediction: Any | None = None
    used_input: Any | None = None
    last_exc: Exception | None = None

    for candidate in candidate_inputs:
        try:
            prediction = model.predict(candidate)
            if len(prediction) != 1:
                raise ValueError(f"Expected single prediction, got {len(prediction)}")
            used_input = candidate
            break
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc

    if prediction is None or used_input is None:
        raise RuntimeError(f"Model prediction failed for all input formats: {last_exc}")

    confidence: float | None = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(used_input)  # type: ignore[attr-defined]
            confidence = float(np.max(proba, axis=1)[0])
        except Exception:
            confidence = None

    return int(prediction[0]), confidence


def _score_with_confidence(comment_texts: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for comment_text in comment_texts:
        pred_int, confidence = _predict_single_text(comment_text)
        results.append(
            {
                "prediction_class": pred_int,
                "prediction_label": SENTIMENT_LABELS.get(pred_int, "unknown"),
                "confidence": confidence,
            }
        )

    return results


def _append_logs(records: list[dict[str, Any]]) -> None:
    logs_path = Path(settings.prediction_logs_path)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = logs_path.exists()
    with logs_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "timestamp_utc",
                "input_comment",
                "prediction_class",
                "prediction_label",
                "confidence",
                "model_version",
                "model_source",
                "status",
                "error",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for record in records:
            writer.writerow(record)


@app.on_event("startup")
def startup_event() -> None:
    global model
    global model_metadata

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    client = MlflowClient()
    model_name = "YouTube_Sentiment_Champion"
    selected_version = None
    selected_source = "alias:champion"
    try:
        selected_version = client.get_model_version_by_alias(model_name, "champion")
        model_uri = f"models:/{model_name}@champion"
    except Exception:
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        if not latest_versions:
            raise RuntimeError("No registered model versions found in MLflow registry.")
        selected_version = sorted(latest_versions, key=lambda mv: int(mv.version), reverse=True)[0]
        model_uri = f"models:/{model_name}/{selected_version.version}"
        selected_source = "version_fallback"

    run = client.get_run(selected_version.run_id)
    metrics = run.data.metrics
    required_macro_f1 = float(settings.min_hand_labeled_macro_f1)
    required_negative_recall = float(settings.min_negative_recall)
    actual_macro_f1 = float(metrics.get("hand_labeled_macro_f1", float("nan")))
    actual_negative_recall = float(metrics.get("hand_labeled_negative_recall", float("nan")))

    if np.isnan(actual_macro_f1) or np.isnan(actual_negative_recall):
        raise RuntimeError(
            "Champion is missing required hand-labeled quality metrics. "
            "Retrain with updated training pipeline before serving."
        )

    if actual_macro_f1 < required_macro_f1 or actual_negative_recall < required_negative_recall:
        raise RuntimeError(
            "Champion does not satisfy serving quality thresholds. "
            f"hand_labeled_macro_f1={actual_macro_f1:.4f} (min {required_macro_f1:.4f}), "
            f"hand_labeled_negative_recall={actual_negative_recall:.4f} (min {required_negative_recall:.4f})."
        )

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model = mlflow.pyfunc.load_model(model_uri)

    model_metadata = {
        "model_name": model_name,
        "model_version": str(selected_version.version),
        "run_id": selected_version.run_id,
        "model_uri": model_uri,
        "model_source": selected_source,
        "hand_labeled_macro_f1": actual_macro_f1,
        "hand_labeled_negative_recall": actual_negative_recall,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_loaded": "true" if model is not None else "false",
        "model_version": str(model_metadata.get("model_version", "unknown")),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    comment = request.comment.strip()
    if not comment:
        raise HTTPException(status_code=422, detail="Comment must not be empty")

    try:
        result = _score_with_confidence([comment])[0]
        _append_logs(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "input_comment": comment,
                    "prediction_class": result["prediction_class"],
                    "prediction_label": result["prediction_label"],
                    "confidence": result["confidence"],
                    "model_version": model_metadata.get("model_version", "unknown"),
                    "model_source": model_metadata.get("model_source", "unknown"),
                    "status": "success",
                    "error": "",
                }
            ]
        )
        return PredictResponse(**result)
    except Exception as exc:  # pylint: disable=broad-except
        _append_logs(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "input_comment": comment,
                    "prediction_class": "",
                    "prediction_label": "",
                    "confidence": "",
                    "model_version": model_metadata.get("model_version", "unknown"),
                    "model_source": model_metadata.get("model_source", "unknown"),
                    "status": "error",
                    "error": str(exc),
                }
            ]
        )
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    cleaned_comments = [comment.strip() for comment in request.comments if comment and comment.strip()]
    if not cleaned_comments:
        raise HTTPException(status_code=422, detail="Batch must include at least one non-empty comment")
    if len(cleaned_comments) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"Batch size exceeds MAX_BATCH_SIZE={settings.max_batch_size}")

    try:
        results = _score_with_confidence(cleaned_comments)
        log_records = []
        for comment, result in zip(cleaned_comments, results):
            log_records.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "input_comment": comment,
                    "prediction_class": result["prediction_class"],
                    "prediction_label": result["prediction_label"],
                    "confidence": result["confidence"],
                    "model_version": model_metadata.get("model_version", "unknown"),
                    "model_source": model_metadata.get("model_source", "unknown"),
                    "status": "success",
                    "error": "",
                }
            )

        _append_logs(log_records)
        return PredictBatchResponse(results=[PredictResponse(**item) for item in results])
    except Exception as exc:  # pylint: disable=broad-except
        _append_logs(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "input_comment": "<batch>",
                    "prediction_class": "",
                    "prediction_label": "",
                    "confidence": "",
                    "model_version": model_metadata.get("model_version", "unknown"),
                    "model_source": model_metadata.get("model_source", "unknown"),
                    "status": "error",
                    "error": str(exc),
                }
            ]
        )
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc
