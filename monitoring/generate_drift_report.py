from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, TargetDriftPreset

from app_config import get_settings


def _resolve_prediction_dataframe(prediction_logs_path: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(prediction_logs_path)
    pred_df = pred_df[pred_df["status"] == "success"].copy()

    pred_df["clean_comment"] = pred_df["input_comment"].astype(str)
    pred_df["sentiment"] = pred_df["prediction_class"].astype("Int64")
    pred_df = pred_df[["clean_comment", "sentiment"]].dropna()
    return pred_df


def main() -> None:
    settings = get_settings()

    baseline_path = Path(settings.baseline_dataset_path)
    prediction_logs_path = Path(settings.prediction_logs_path)
    reports_dir = Path("monitoring") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline dataset not found: {baseline_path}")
    if not prediction_logs_path.exists():
        raise FileNotFoundError(f"Prediction logs not found: {prediction_logs_path}")

    baseline_df = pd.read_parquet(baseline_path)
    baseline_df = baseline_df[["clean_comment", "sentiment"]].dropna().copy()

    current_df = _resolve_prediction_dataframe(prediction_logs_path)

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=baseline_df, current_data=current_df)
    data_drift_report.save_html(str(reports_dir / "data_drift_report.html"))

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=baseline_df, current_data=current_df)
    target_drift_report.save_html(str(reports_dir / "target_drift_report.html"))

    print("Generated reports:")
    print(reports_dir / "data_drift_report.html")
    print(reports_dir / "target_drift_report.html")


if __name__ == "__main__":
    main()
