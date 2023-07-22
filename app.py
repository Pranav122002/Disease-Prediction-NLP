import numpy as np
import pickle
import pandas as pd
import streamlit as st
import zipfile
import os


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


zip_file_path = "dataset-zip.zip"
extraction_path = os.getcwd()
unzip_file(zip_file_path, extraction_path)

model = pickle.load(open("models/tfidf_trigrams_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizers/tfidf_vectorizer3.pkl", "rb"))

df = pd.read_csv("dataset/drugsComTrain_raw.csv")
df_drug = df[(df["rating"] >= 9) & (df["usefulCount"] >= 100)].sort_values(
    by=["rating", "usefulCount"], ascending=[False, False]
)


def recommend_drug(disease):
    recommended_drug_list = (
        df_drug[df_drug["condition"] == disease]["drugName"].head(3).tolist()
    )
    return recommended_drug_list


def main():
    st.title("Disease Prediction")

    st.sidebar.markdown(
        "The **PassiveAggressiveClassifier** is a popular algorithm for online learning tasks and is often used for binary text classification problems. It is a type of linear classifier, suitable for large-scale learning. The PassiveAggressiveClassifier model, is trained on the TF-IDF transformed training data."
    )
    st.sidebar.markdown(
        "Term Frequency-Inverse Document Frequency (TF-IDF). **TfidfVectorizer** is used to convert the text data into numerical features."
    )
    st.sidebar.markdown(
        "**Dataset** includes 884 diseases, but the model is trained only on 11 most common diseases as to increase the accuracy and as this diseases contains highest instances in the dataset."
    )
    st.sidebar.markdown(
        "Diseases on which model is trained are:\n"
        "- Birth Control\n"
        "- Depression\n"
        "- Anxiety\n"
        "- Acne\n"
        "- Insomnia\n"
        "- Diabetes, Type 2\n"
        "- High Blood Pressure\n"
        "- Migraine\n"
        "- Constipation\n"
        "- Cough\n"
        "- GERD\n"
    )

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
