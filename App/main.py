import streamlit as st
import joblib
import pandas as pd
from preprocessing import TextPreprocessor
import sys
import os

# Add parent directory to Python path so it can find preprocessing.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import your classes
from preprocessing import TextPreprocessor, Explore, Clean

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Streamlit UI
st.set_page_config(page_title="Symptom-Based Disease Predictor", layout="centered")
st.title("🩺 Symptom-Based Disease Prediction")

st.write("Enter your symptoms below:")

# Input text box
user_input = st.text_area("Symptoms", height=150)

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        # Preprocess input
        temp_df = pd.DataFrame({"text": [user_input]})
        temp_df = preprocessor.preprocess(temp_df, "text")
        doc = temp_df['document'].values[0]

        # Vectorize
        vect_input = vectorizer.transform([doc])

        # Predict
        pred_code = model.predict(vect_input)[0]

        # Map code to label
        # Load label encoder from your notebook or manually create a mapping
        label_mapping = {
            0: 'Acne', 1: 'Allergy', 2: 'Arthritis', 3: 'Bronchial Asthma', 4: 'Chicken pox',
            5: 'Common Cold', 6: 'Cervical spondylosis', 7: 'Dengue', 8: 'Diabetes', 9: 'Dimorphic Hemorrhoids',
            10: 'Drug reaction', 11: 'Fungal infection', 12: 'Gastroesophageal reflux disease',
            13: 'Hypertension', 14: 'Impetigo', 15: 'Jaundice', 16: 'Malaria', 17: 'Migraine',
            18: 'Peptic ulcer disease', 19: 'Pneumonia', 20: 'Psoriasis', 21: 'Typhoid',
            22: 'Urinary tract infection', 23: 'Varicose Veins'
        }

        st.success(f"Predicted Disease: **{label_mapping[pred_code]}**")
