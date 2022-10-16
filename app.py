import pandas as pd
import streamlit as st

import pickle 

def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

st.title("APLIKASI LINGUISTIK FORENSIK BERBASIS KORPUS KASUS HUKUM UJARAN KEBENCIAN")

df = pd.read_csv('tweetclean.csv')
pkl_filename = "LR_Model.pkl"

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

kalimat = df['tweet'].values.tolist() #for saving old shape
kalimat2 = kalimat.copy() #df with new shape with new input

kalimat2.append(sentence) #add input text to dataset
sentences_preprocessing = [sent.lower().split(" ") for sent in kalimat2]#preprocessing

similarity_result = []#variable for similarity result

#calculate similarity score
for i in range(len(kalimat2)-1):
  similarity_result.append(jaccard_similarity(sentences_preprocessing[-1], sentences_preprocessing[i]))
  
#create Data Frame tweet and similarity score
dict = {'sentences': kalimat, 'similarity_score': similarity_result} 
df_similarity = pd.DataFrame(dict)
df2 = df_similarity.sort_values(by='similarity_score', ascending=False).head() #data frame with similarity score between input and dataset

products_list = df2.values.tolist() #convert to list

if sentence:
    st.text("Hasil:")
    st.write(prediction[0])

    st.subheader('Kelas Label dan Nomor Indeks')
    st.write(classifier.classes_)
    
    st.subheader("Hasil Similariti Kalimat:")
    for i in range(1, len(products_list)+1):
        st.write(i, " ", products_list[i-1][0])
