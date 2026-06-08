import streamlit as st

from predictor import get_predictions

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠"
)

st.title("🧠 Next Word Prediction")

text = st.text_input(
    "Enter text"
)

if text:

    with st.spinner(
        "Generating predictions..."
    ):
        predictions = get_predictions(text)

    st.subheader("Top Predictions")

    for rank, (word, prob) in enumerate(predictions, start=1):
        st.metric(
            label=f"Suggestion #{rank}",
            value=word,
            delta=f"{prob:.2%}"
        )