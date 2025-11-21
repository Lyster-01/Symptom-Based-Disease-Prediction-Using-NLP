import streamlit as st
import pickle
import pandas as pd
from __init__ import TextPreprocessor   # your preprocessing class

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load label encoder (if you saved it)
try:
    with open("model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except:
    le = None

# Preprocessor
preprocessor = TextPreprocessor()

# Streamlit App UI
st.title("🩺 Symptom-Based Disease Prediction Using NLP")

st.write(
    "Enter your symptoms below. The system will analyze them and predict the most likely disease."
)

# Text Input
symptoms = st.text_area("Enter symptoms here:")

# Predict Button
if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        # Convert user input into a dataframe
        df_input = pd.DataFrame({"text": [symptoms]})

        # Preprocess text (clean → tokenize → lemmatize → join)
        df_processed = preprocessor.preprocess(df_input, "text")

        # Vectorize using saved vectorizer
        X_input = vectorizer.transform(df_processed["document"])

        # Predict
        pred_code = model.predict(X_input)[0]

        # Decode label if label encoder exists
        if le:
            predicted_disease = le.inverse_transform([pred_code])[0]
        else:
            predicted_disease = pred_code

        st.success(f"### 🧬 Predicted Disease: **{predicted_disease}**")
