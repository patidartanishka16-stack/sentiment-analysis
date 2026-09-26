import streamlit as st
import joblib
import re

# Load trained pipeline
model = joblib.load("models/sentiment_pipeline.pkl")


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 AI-Powered Sentiment Analysis")
st.write("Enter a customer review and analyze its sentiment.")


# User input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: The product is amazing and I really love it."
)


# Analyze button
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)

        # Prediction
        prediction = model.predict([cleaned_review])[0]

        # Probability
        probabilities = model.predict_proba([cleaned_review])[0]
        confidence = probabilities.max()

        probability_dict = dict(
            zip(model.classes_, probabilities)
        )

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success(f"😊 {prediction}")

        elif prediction == "Negative":
            st.error(f"😞 {prediction}")

        else:
            st.info(f"😐 {prediction}")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("Class Probabilities")

        for sentiment, probability in probability_dict.items():
            st.write(
                f"**{sentiment}:** {probability * 100:.2f}%"
            )