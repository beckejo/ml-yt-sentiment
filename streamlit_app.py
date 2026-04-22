import pandas as pd
import requests
import streamlit as st

from app_config import get_settings


settings = get_settings()
API_BASE_URL = settings.api_base_url.rstrip("/")


def call_predict(comment: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"comment": comment},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def call_predict_batch(comments: list[str]) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/predict_batch",
        json={"comments": comments},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


st.title("YouTube Comment Sentiment Analyzer")
st.write("Analyze single comments or upload a CSV for batch sentiment inference.")
st.caption(f"Connected API endpoint: {API_BASE_URL}")

single_tab, batch_tab = st.tabs(["Single Comment", "Batch CSV Upload"])

with single_tab:
    user_input = st.text_area("Paste a YouTube comment", height=120)
    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a comment before requesting a prediction.")
        else:
            try:
                payload = call_predict(user_input.strip())
                st.success("Prediction received")
                st.json(payload)
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")

with batch_tab:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            candidate_columns = ["comment", "clean_comment", "text", "body"]
            comment_column = next((col for col in candidate_columns if col in batch_df.columns), None)

            if comment_column is None:
                st.error("CSV must include one of these columns: comment, clean_comment, text, body")
            else:
                comments = batch_df[comment_column].dropna().astype(str).str.strip()
                comments = comments[comments != ""].tolist()

                if st.button("Run Batch Prediction"):
                    if not comments:
                        st.warning("No valid comments were found in the uploaded file.")
                    else:
                        try:
                            payload = call_predict_batch(comments)
                            results = payload.get("results", [])
                            result_df = pd.DataFrame(results)
                            if not result_df.empty:
                                result_df.insert(0, "comment", comments[: len(result_df)])
                            st.success(f"Received {len(results)} predictions")
                            st.dataframe(result_df, use_container_width=True)
                        except requests.RequestException as exc:
                            st.error(f"Batch API request failed: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to process uploaded file: {exc}")