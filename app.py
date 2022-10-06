import streamlit as st
import pickle
import numpy as np
import base64


def get_base64(bin_file):
    with open(bin_file, 'rb') as f1:
        data = f1.read()
    return base64.b64encode(data).decode()

#background-img
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./emotion.png')


def predict_emotion(mn1,tf,text):
  text = [text]
  cvect = tf.transform(text).toarray()
  pred = mn1.predict(cvect)
  return pred

def my_tokenizer(s):
  return s.split(' ')

file1 = open('./model/model268.pkl','rb')
file2 = open('./model/tfidf.pkl','rb')

#file1=open("./model268.pkl","rb")
#file2=open("./tfidf.pkl","rb")

mn1=pickle.load(file1)
tf=pickle.load(file2)

file1.close()
file2.close()

def predict_emotion1(text):
  pred = predict_emotion(mn1,tf,text)
  prediction  = int(pred)
  if prediction ==0:
    prediction = 'Angry'
  elif prediction == 1:
    prediction ='Happy'
  elif prediction == 2:
    prediction = 'Sad'
  else:
    prediction = 'Neutral'
  return prediction

st.write('  ')
st.markdown('##')
st.write('  ')
st.write('# Emotion Detection from text')

text=st.text_area(" Enter the text in Hindi to predict the emotion")

predict_btn=st.button("predict_emotion")

if predict_btn:
    result=predict_emotion1(text)
    st.write('## Emotion behind the sentence is',result)
    #st.image(recommended_movie_posters[0])