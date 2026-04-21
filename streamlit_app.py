import streamlit as st
import mlflow.pyfunc
import pandas as pd

st.title("YouTube Comment Sentiment Analyzer")
st.write("Paste a YouTube comment to see if our champion model thinks it is Positive, Neutral, or Negative!")

# Load the champion model from the MLflow registry
@st.cache_resource
def load_model():
    model_uri = "models:/YouTube_Sentiment_Champion/latest"
    return mlflow.pyfunc.load_model(model_uri)

try:
    model = load_model()
    st.success("Champion model loaded successfully from MLflow Registry!")
except Exception as e:
    st.error(f"Error loading model: {e}")

user_input = st.text_area("YouTube Comment:")

if st.button("Predict Sentiment"):
    if user_input:
        st.info("Model prediction logic would execute here!")
        # To actually predict:
        # prediction = model.predict(pd.DataFrame([{"clean_comment": user_input}]))
        # st.write(f"Predicted Sentiment Class: {prediction}")
    else:
        st.warning("Please enter a comment first.")