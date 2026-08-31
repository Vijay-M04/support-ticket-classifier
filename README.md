# 🎫 AI Support Ticket Category Classifier

An NLP-based machine learning application that automatically
classifies customer support tickets into appropriate categories
and detects the urgency level of each ticket.

## 🚀 Features

- Customer support ticket classification
- Automatic category prediction
- Urgency detection: Low, Medium, High
- TF-IDF text vectorization
- Logistic Regression classification
- Model confidence score
- Interactive Streamlit web application

## 🧠 Machine Learning Approach

The project uses:

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

### Workflow

Customer Ticket
→ Text Processing
→ TF-IDF Vectorization
→ Logistic Regression
→ Category Prediction

A keyword-based rule system is used separately
to determine ticket urgency.

## 📂 Project Structure

```text
support-ticket-classifier/
│
├── data/
│   └── tickets.csv
│
├── model/
│   └── ticket_classifier.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
