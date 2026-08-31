import streamlit as st
import pickle


# -----------------------------
# Load trained model
# -----------------------------
with open("model/ticket_classifier.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------
# Urgency Detection
# -----------------------------
def detect_urgency(text):

    text = text.lower()

    high_keywords = [
        "urgent",
        "immediately",
        "emergency",
        "critical",
        "fraud",
        "stolen",
        "locked"
    ]

    medium_keywords = [
        "soon",
        "delayed",
        "problem",
        "issue",
        "error",
        "failed",
        "cannot",
        "unable"
    ]

    for word in high_keywords:
        if word in text:
            return "High"

    for word in medium_keywords:
        if word in text:
            return "Medium"

    return "Low"


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)


# -----------------------------
# Header
# -----------------------------
st.title("🎫 AI Support Ticket Classifier")

st.write(
    "An NLP-based application that automatically "
    "classifies customer support tickets and detects urgency."
)

st.divider()


# -----------------------------
# Ticket Input
# -----------------------------
st.subheader("📩 Enter Customer Ticket")

ticket = st.text_area(
    "Support Ticket",
    placeholder=(
        "Example: My payment was charged twice "
        "and I need a refund immediately."
    ),
    height=150
)


# -----------------------------
# Classification
# -----------------------------
if st.button("🔍 Classify Ticket", use_container_width=True):

    if not ticket.strip():

        st.warning("⚠️ Please enter a support ticket.")

    else:

        # Category prediction
        prediction = model.predict([ticket])[0]

        # Confidence score
        probabilities = model.predict_proba([ticket])[0]
        confidence = max(probabilities) * 100

        # Urgency prediction
        urgency = detect_urgency(ticket)

        st.divider()

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Category",
                prediction
            )

        with col2:
            st.metric(
                "Urgency",
                urgency
            )

        with col3:
            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

        st.success(
            f"Ticket classified as **{prediction}** "
            f"with **{urgency} urgency**."
        )


# -----------------------------
# About
# -----------------------------
with st.expander("ℹ️ About this Project"):

    st.write(
        """
        **Machine Learning:** Logistic Regression

        **NLP Technique:** TF-IDF Vectorization

        **Urgency Detection:** Keyword-based rules

        **Framework:** Streamlit

        The model is trained on labelled customer support
        ticket data and predicts the appropriate ticket category.
        """
    )