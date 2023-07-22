# Disease-Prediction-NLP
Disease prediction based on symptoms provided in a single text input using Natural Language Processing (NLP).

## Demo
* https://disease-prediction-nlp.streamlit.app/

## Description

The **PassiveAggressiveClassifier** is a popular algorithm for online learning tasks and is often used for binary text classification problems. It is a type of linear classifier, suitable for large-scale learning. The PassiveAggressiveClassifier model, is trained on the TF-IDF transformed training data.

Term Frequency-Inverse Document Frequency (TF-IDF). **TfidfVectorizer** is used to convert the text data into numerical features.

**Dataset** includes 884 diseases, but the model is trained only on 11 most common diseases as to increase the accuracy and as this diseases contains highest instances in the dataset.

Diseases on which model is trained are :
- Birth Control          
- Depression              
- Anxiety                 
- Acne                    
- Insomnia                
- Diabetes, Type 2        
- High Blood Pressure     
- Migraine                
- Constipation            
- Cough                    
- GERD                     

## Installation & Running
1. Install the required packages:
```
pip install streamlit numpy pandas scikit-learn nltk matplotlib
```

2. Compile and save the models by running the entire disease_prediction.ipynb file.

3. Run the Streamlit app:
```
streamlit run app.py
```
