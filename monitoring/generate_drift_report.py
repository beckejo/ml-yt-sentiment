from pathlib import Path
import sys

# Make project-root imports work no matter where this script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pandas.errors import ParserError
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

from app_config import get_settings


def _resolve_prediction_dataframe(prediction_logs_path: Path) -> pd.DataFrame:
    try:
        pred_df = pd.read_csv(prediction_logs_path)
    except ParserError:
        # Skip malformed rows so monitoring can proceed even if a few log lines are bad.
        pred_df = pd.read_csv(prediction_logs_path, engine="python", on_bad_lines="skip")

    required_columns = {"status", "input_comment", "prediction_class"}
    missing_columns = required_columns.difference(pred_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in prediction logs: {sorted(missing_columns)}")

    pred_df = pred_df[pred_df["status"] == "success"].copy()

    pred_df["clean_comment"] = pred_df["input_comment"].astype(str)
    pred_df["sentiment"] = pd.to_numeric(pred_df["prediction_class"], errors="coerce")
    pred_df = pred_df[["clean_comment", "sentiment"]].dropna()
    pred_df["sentiment"] = pred_df["sentiment"].astype("int64")
    if pred_df.empty:
        raise ValueError("No valid successful predictions found to compute drift report.")

    return pred_df


def _report_has_widgets(report: Report) -> bool:
    report_dict = report.as_dict()
    return bool(report_dict.get("widgets"))


def main() -> None:
    settings = get_settings()

    baseline_path = Path(settings.baseline_dataset_path)
    prediction_logs_path = Path(settings.prediction_logs_path)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    if not prediction_logs_path.is_absolute():
        prediction_logs_path = PROJECT_ROOT / prediction_logs_path

    reports_dir = PROJECT_ROOT / "monitoring" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline dataset not found: {baseline_path}")
    if not prediction_logs_path.exists():
        raise FileNotFoundError(f"Prediction logs not found: {prediction_logs_path}")

    baseline_df = pd.read_parquet(baseline_path)
    baseline_df = baseline_df[["clean_comment", "sentiment"]].dropna().copy()
    baseline_df["clean_comment"] = baseline_df["clean_comment"].astype(str)
    baseline_df["sentiment"] = pd.to_numeric(baseline_df["sentiment"], errors="coerce")
    baseline_df = baseline_df.dropna(subset=["sentiment"])
    baseline_df["sentiment"] = baseline_df["sentiment"].astype("int64")

    current_df = _resolve_prediction_dataframe(prediction_logs_path)

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=baseline_df, current_data=current_df)
    data_drift_report.save_html(str(reports_dir / "data_drift_report.html"))

    target_report_path = reports_dir / "target_drift_report.html"
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=baseline_df, current_data=current_df)

    if _report_has_widgets(target_drift_report):
        target_drift_report.save_html(str(target_report_path))
    else:
        # Some Evidently 0.6.x environments emit an empty dashboard for TargetDriftPreset.
        # Fall back to a direct column drift metric so the HTML is always readable.
        fallback_target_report = Report(metrics=[ColumnDriftMetric(column_name="sentiment")])
        fallback_target_report.run(reference_data=baseline_df, current_data=current_df)
        fallback_target_report.save_html(str(target_report_path))

    print("Generated reports:")
    print(reports_dir / "data_drift_report.html")
    print(target_report_path)


if __name__ == "__main__":
    main()
