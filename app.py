import numpy as np
import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open("models/tfidf_bigrams_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizers/tfidf_vectorizer2.pkl", "rb"))

df = pd.read_csv('dataset/drugsComTrain_raw.csv')
df_drug = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])


def recommend_drug(disease):
    recommended_drug_list = df_drug[df_drug['condition'] == disease]['drugName'].head(3).tolist()
    return recommended_drug_list


def main():
    st.title("Disease Prediction")

    with st.form(key="disease_clf_form"):
        raw_text = st.text_area("Type Your Symptoms Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:

        text_transformed = vectorizer.transform([raw_text])
        result = model.predict(text_transformed)[0]
        recommended_drug_list = recommend_drug(result)

        col1, col2 = st.columns(2)
        col1.header("Predicted Disease:")
        col2.header("Recommended Drugs:")

        col1.success(result)

        if recommended_drug_list:
            for drug in recommended_drug_list:
                col2.write(drug)
        else:
            col2.write("No recommended drugs found for the predicted disease.")


if __name__ == "__main__":
    main()
