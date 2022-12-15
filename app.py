import pandas as pd
import streamlit as st

import pickle 

def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

st.set_page_config(page_title="Si-Yudistria: Sistem Aplikasi Yuridis Deteksi Ujaran")

st.title("Si-Yudistria")
st.header("Sistem Aplikasi Yuridis Deteksi Ujaran")

df = pd.read_csv('dummyhatespeech.csv')
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
df2 = df.copy()
df2['similarity_score'] = df_similarity['similarity_score']
df2 = df2.sort_values(by='similarity_score', ascending=False).reset_index().head(3) #data frame with similarity score between input and dataset
df2 = df2.drop(['index', 'Unnamed: 0'], axis=1)
products_list = df2.values.tolist() #convert to list

#st.write(df2.head())

if sentence:
    st.subheader("Hasil prediksi:")
    st.write(prediction[0])

    ##st.subheader('Kelas Label dan Nomor Indeks')
    #st.write(classifier.classes_)
    
    st.subheader("Hasil Similariti Kalimat:")
    for i in range(1, len(products_list)+1):
        st.write(i, " ", products_list[i-1][1])
        with st.expander("see more"):
          #st.write("Kalimat: ",df2["sentences"].iloc[i-1])
          st.write("**Kata Kunci:** ", df2["katakunci"].iloc[i-1])
          st.write("**Pasal Sangkaan:** ",  df2["pasal"].iloc[i-1])
          st.write("**Status Perkara:** ", df2["status"].iloc[i-1])
          st.write("**Kronologis:** ", df2["kronologis"].iloc[i-1])
          st.write("**Agama:** ", df2["agama"].iloc[i-1])
          st.write("**Jenis Kelamin:** ",  df2["gender"].iloc[i-1])
          
          
          
        

    
