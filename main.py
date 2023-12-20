from flask import Flask, render_template, request
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pickle import load
from scipy import sparse
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)

model = pickle.load(open('nbmodel.pkl','rb'))
stopwords_list = stopwords.words('english')
vectorizer=load(open('tfidf.pkl','rb'))

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('“','',text)
    text = re.sub('”','',text)
    text = re.sub('’','',text)
    text = re.sub('–','',text)
    text = re.sub('‘','',text)

    return text

def contains_question(headline):
    if "?" in headline or headline.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do')):
        return 1
    else:
        return 0

def contains_exclamation(headline):
    if "!" in headline:
        return 1
    else:
        return 0

def starts_with_num(headline):
    if headline.startswith(('1','2','3','4','5','6','7','8','9')):
        return 1
    else:
        return 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def check():
    sentence = request.form.get("headline")

    cleaned_sentence = clean_text_round1(sentence)
    headline_words = len(cleaned_sentence.split())
    question = contains_question(cleaned_sentence)
    exclamation = contains_exclamation(cleaned_sentence)
    with_num = starts_with_num(cleaned_sentence)
    input=[cleaned_sentence]
    vectorized = vectorizer.transform(input)
    final = sparse.hstack([question,exclamation,with_num,headline_words,vectorized])
    result = model.predict(final)

    if result == 1:
        return render_template('index.html', label=1, headline=sentence)
    else:
        return render_template('index.html', label=0, headline=sentence)

if __name__ == "__main__":
    app.run(debug=True)