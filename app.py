from flask import Flask, render_template
import flask
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import praw

app = Flask(__name__)

def text_preprocess(text):
    text = re.sub(r'[^\w\s]', '', text) 
    l_text = " ".join(word for word in text.lower().split() if word not in ENGLISH_STOP_WORDS)

    return l_text

with open('encoding.pkl', "rb") as f:
    enc = pickle.load(f)
    
with open('kn.pkl', "rb") as f:
    knn = pickle.load(f)
    
with open('senti.pkl', "rb") as f:
    sid = pickle.load(f)
    
df_tokens = pd.read_csv('tokenizer.csv')
embedding_matrix = np.array(pd.read_csv('glove.csv', sep=' '))


def data_from_url(url):
    data = {}
    reddit = praw.Reddit(client_id='rGGcUZbUNTiCFw',
                   client_secret='hD5kq4AUUN4qhvLLrqO77B0FvAGseQ', 
                   user_agent='Reddit WebScrapping')

    sub_data = reddit.submission(url=str(url))
    
    data['Title'] = [str(sub_data.title)]
    data['Gilded'] = [sub_data.gilded]
    data['Over_18'] = [sub_data.over_18]
    data['Number_of_Comments'] = [sub_data.num_comments]
    scores = sid.polarity_scores(sub_data.title)
    compound = scores['compound']
    
    if (compound >= 0.5):
        data['Predicted_value'] = ['positive']
    
    elif (compound >= 0) & (compound <= 0.5):
        data['Predicted_value'] = ['neutral']

    elif (compound <= 0):
        data['Predicted_value'] = ['negative']
        
    df = pd.DataFrame(data)
        
    return df
    
            
@app.route('/')
def home():
   return render_template('index.html')
            
@app.route('/predict', methods=['POST'])
def predict():
   url = str(flask.request.form['url'])
   data = data_from_url(url)
   title = text_preprocess(data['Title'][0])
   test_title = []
   for word in title.split():
       if word in df_tokens.columns:
           test_title.append(df_tokens[word])
   maxlen = 300
   test_title = test_title + [0] * (maxlen - len(test_title))
   vectors = []
   for n in test_title:
       vectors.append(embedding_matrix[n])
   vectors = [item for sublist in vectors for item in sublist]
   arr = np.array(vectors)
   final_vector = np.mean(arr, axis=0)
   final_vector = pd.DataFrame(np.array(final_vector)).T
   categories = ['Over_18', 'Predicted_value']
   test_encoded = enc.transform(data[categories])
   data.drop(["Title", 'Over_18', 'Predicted_value'], axis=1, inplace=True)
   data.reset_index(inplace=True, drop=True)
   col_names = [False, True, 'negative', 'neutral', 'positive']
   test = pd.DataFrame(test_encoded.todense(), columns=col_names)
   X_test = pd.concat([data, final_vector, test], axis=1)
   score = int(knn.predict(X_test))
   
   return render_template('index.html', score='Score: {}'.format(score))

if __name__ == "__main__":
   app.run(debug=True)
