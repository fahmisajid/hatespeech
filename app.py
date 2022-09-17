import pandas as pd
import streamlit as st

import pickle 

st.title("APLIKASI LINGUISTIK FORENSIK BERBASIS KORPUS KASUS HUKUM UJARAN KEBENCIAN")

pkl_filename = "Lr_model.pkl"

with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

with open("countVectLR", 'rb') as file:
    count_vect = pickle.load(file)

with open("tfidfLR", 'rb') as file:
    tfidf_transformer = pickle.load(file)

sentence = st.text_input('Masukkan Kalimat:') 

text_new =[sentence]
X_new_counts = count_vect.transform(text_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = classifier.predict(X_new_tfidf)
prediction_proba = classifier.predict_proba(X_new_tfidf)


if sentence:
    st.text("Hasil:")
    st.write(prediction[0])

    st.subheader('Kelas Label dan Nomor Indeks')
    st.write(classifier.classes_)