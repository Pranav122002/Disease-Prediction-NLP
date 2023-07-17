import numpy as np
import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open("models/tfidf_trigrams_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizers/tfidf_vectorizer.pkl", "rb"))

def main():
    st.title("Disease Prediction")

    with st.form(key="disease_clf_form"):
        raw_text = st.text_area("Type Your Symptoms Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        st.success("Predicted Disease : ")

        text_transformed = vectorizer.transform([raw_text])  
        result = model.predict(text_transformed)[0]
        st.write(result)

if __name__ == "__main__":
    main()
