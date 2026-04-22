from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import great_expectations as gx
import pandas as pd

from app_config import get_settings
from dataops_utils import ingest_comments_for_videos, ingest_video_ids, ingest_video_stats


REQUIRED_COLUMNS = [
    "video_id",
    "views",
    "likes",
    "comments",
    "likes_per_100_views",
    "sentiment",
    "comment",
    "clean_comment",
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _map_sentiment(raw_value: object) -> int | None:
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


def load_reddit_dataset(dataset_path: str) -> pd.DataFrame:
    if not dataset_path:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Reddit dataset path does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        reddit_df = pd.read_parquet(path)
    else:
        reddit_df = pd.read_csv(path)

    comment_column_candidates = ["comment", "clean_comment", "text", "body"]
    sentiment_column_candidates = ["sentiment", "label", "target"]

    comment_col = next((c for c in comment_column_candidates if c in reddit_df.columns), None)
    sentiment_col = next((c for c in sentiment_column_candidates if c in reddit_df.columns), None)

    if comment_col is None or sentiment_col is None:
        raise ValueError(
            "Reddit dataset must include a comment column (comment/clean_comment/text/body) "
            "and sentiment column (sentiment/label/target)."
        )

    out_df = pd.DataFrame()
    out_df["comment"] = reddit_df[comment_col].astype(str)
    out_df["clean_comment"] = out_df["comment"].apply(clean_text)
    out_df["sentiment"] = reddit_df[sentiment_col].apply(_map_sentiment)

    if "likes" in reddit_df.columns:
        likes_raw = reddit_df["likes"]
    elif "score" in reddit_df.columns:
        likes_raw = reddit_df["score"]
    else:
        likes_raw = pd.Series([0] * len(reddit_df), index=reddit_df.index)

    if "views" in reddit_df.columns:
        views_raw = reddit_df["views"]
    else:
        views_raw = pd.Series([100] * len(reddit_df), index=reddit_df.index)

    if "comments" in reddit_df.columns:
        comments_raw = reddit_df["comments"]
    elif "num_comments" in reddit_df.columns:
        comments_raw = reddit_df["num_comments"]
    else:
        comments_raw = pd.Series([1] * len(reddit_df), index=reddit_df.index)

    likes_series = pd.to_numeric(likes_raw, errors="coerce").fillna(0)
    views_series = pd.to_numeric(views_raw, errors="coerce").fillna(100)
    comments_series = pd.to_numeric(comments_raw, errors="coerce").fillna(1)

    out_df["likes"] = likes_series.astype("int64")
    out_df["views"] = views_series.astype("int64")
    out_df["comments"] = comments_series.astype("int64")
    out_df["likes_per_100_views"] = (out_df["likes"] / out_df["views"].replace(0, 1) * 100).round(3)
    out_df["video_id"] = [f"reddit_{idx}" for idx in out_df.index]
    out_df["source"] = "reddit"

    out_df = out_df.dropna(subset=["sentiment"])
    out_df["sentiment"] = out_df["sentiment"].astype("int64")
    return out_df


def build_youtube_dataset() -> pd.DataFrame:
    settings = get_settings()
    if not (settings.videos_api_key and settings.comments_api_key and settings.stats_api_key):
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    video_ids = ingest_video_ids(settings.videos_api_key)
    if not video_ids:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    video_ids_df = pd.DataFrame(video_ids)
    if video_ids_df.empty or "video_id" not in video_ids_df.columns:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    stats = ingest_video_stats(list(video_ids_df["video_id"].astype(str)), settings.stats_api_key)
    stats_df = pd.DataFrame(stats)
    if stats_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    for col in ["views", "likes", "comments"]:
        stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    stats_df = stats_df.dropna(subset=["video_id", "views", "likes", "comments"])
    stats_df[["views", "likes", "comments"]] = stats_df[["views", "likes", "comments"]].astype("int64")
    stats_df = stats_df[stats_df["comments"] >= 10].copy()

    stats_df["likes_per_100_views"] = (stats_df["likes"] / stats_df["views"].replace(0, 1) * 100).round(3)
    stats_df = stats_df.dropna(subset=["likes_per_100_views"])

    if stats_df["likes_per_100_views"].nunique() < 3:
        ranked = stats_df["likes_per_100_views"].rank(method="first")
        stats_df["sentiment"] = pd.qcut(ranked, q=3, labels=[0, 1, 2]).astype("int64")
    else:
        stats_df["sentiment"] = pd.qcut(
            stats_df["likes_per_100_views"],
            q=3,
            labels=[0, 1, 2],
        ).astype("int64")

    comments_df = ingest_comments_for_videos(settings.comments_api_key, list(stats_df["video_id"]))
    if comments_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    comments_agg_df = comments_df.groupby("video_id")["comment"].agg(lambda x: " ".join(x)).reset_index()
    comments_agg_df["clean_comment"] = comments_agg_df["comment"].apply(clean_text)

    merged_df = stats_df.merge(comments_agg_df, how="inner", on="video_id")
    merged_df["source"] = "youtube"
    return merged_df


def merge_sources(youtube_df: pd.DataFrame, reddit_df: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [youtube_df, reddit_df] if frame is not None and not frame.empty]
    if not frames:
        raise ValueError("No data available. Provide YouTube API keys and/or a Reddit dataset path.")

    complete_data = pd.concat(frames, ignore_index=True)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in complete_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns before validation: {missing_cols}")

    complete_data["video_id"] = complete_data["video_id"].astype(str)
    complete_data["comment"] = complete_data["comment"].astype(str)
    complete_data["clean_comment"] = complete_data["clean_comment"].astype(str)

    for col in ["views", "likes", "comments", "sentiment"]:
        complete_data[col] = pd.to_numeric(complete_data[col], errors="coerce")

    complete_data["likes_per_100_views"] = pd.to_numeric(complete_data["likes_per_100_views"], errors="coerce")
    complete_data = complete_data.dropna(subset=REQUIRED_COLUMNS)

    complete_data[["views", "likes", "comments", "sentiment"]] = (
        complete_data[["views", "likes", "comments", "sentiment"]].astype("int64")
    )
    complete_data["likes_per_100_views"] = complete_data["likes_per_100_views"].astype(float)
    return complete_data


def validate_with_great_expectations(complete_data: pd.DataFrame) -> None:
    context = gx.get_context()

    try:
        data_source = context.data_sources.get("pandas")
    except Exception:
        data_source = context.data_sources.add_pandas("pandas")

    batch_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    data_asset = data_source.add_dataframe_asset(name=f"youtube_video_data_{batch_suffix}")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": complete_data})

    suite = gx.ExpectationSuite(name=f"youtube_video_data_expectations_{batch_suffix}")
    suite = context.suites.add(suite)

    for col in REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    type_expectations = {
        "video_id": "object",
        "views": "int64",
        "likes": "int64",
        "comments": "int64",
        "likes_per_100_views": "float",
        "sentiment": "int64",
        "comment": "object",
        "clean_comment": "object",
    }
    for col, dtype in type_expectations.items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column=col, type_=dtype))

    for col in REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    validation_results = batch.validate(suite)
    if not validation_results.success:
        raise ValueError("Great Expectations validation failed for processed dataset.")


def persist_dataset(complete_data: pd.DataFrame, output_path: str) -> None:
    complete_data.to_parquet(output_path, index=False)


def main() -> None:
    settings = get_settings()

    youtube_df = build_youtube_dataset()
    reddit_df = load_reddit_dataset(settings.reddit_dataset_path)
    complete_data = merge_sources(youtube_df, reddit_df)

    validate_with_great_expectations(complete_data)
    persist_dataset(complete_data, settings.data_output_path)

    print(f"Dataset rows: {len(complete_data)}")
    print(f"Saved dataset to: {settings.data_output_path}")


if __name__ == "__main__":
    main()