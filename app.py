import numpy as np
import pickle
import pandas as pd
import streamlit as st 

model = pickle.load(open('RandomForest.pickle', 'rb'))

def main():
    st.title('Disease Prediction')

    with st.form(key='disease_clf_form'):
        raw_text = st.text_area('Type Your Symptomns Here')
        submit_text = st.form_submit_button(label='Submit')
    
    if submit_text:

        col1, col2 = st.columns(2)

        with col1:
            st.success('Symptomns')
            st.write(raw_text)

        with col2:
            st.success('Predictions')
            
            result = model.predict([raw_text])
            st.write(result)

if __name__ == '__main__':
    main()