import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string


rawData = []
Y = set(stopwords.words('english'))
real_review = []
fake_review = []
real_review_length = []
fake_review_length = []
real_word_length = []
fake_word_length = []
real_uppers = []
fake_uppers = []
real_stopwords = []
fake_stopwords = []
real_punc = []
fake_punc = []
real_contain_title = []
fake_contain_title = []
real_title = []
fake_title = []

def loadData(path, Text=None):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label, Review_Title) = parseReview(line)
            rawData.append((Id, Text, Label, Review_Title))
            
def parseReview(reviewLine):
    return (int(reviewLine[0]), reviewLine[8], 'fake' if reviewLine[1] == '__label1__' else 'real', reviewLine[6])

loadData('amazon_reviews.txt')

for i in range(0, len(rawData)):
    if(rawData[i][2] == 'real'):
        real_review_length.append(len(rawData[i][1]))
    else:
        fake_review_length.append(len(rawData[i][1]))
        
for i in range(0, len(rawData)):
    if(rawData[i][2] == 'real'):
        real_review.append(rawData[i][1])
        real_title.append(rawData[i][3])
    else:
        fake_review.append(rawData[i][1])
        fake_title.append(rawData[i][3])

for i in range(0, len(real_review)):
    word_length = [len(x) for x in real_review[i].split()]
    real_word_length.append(np.mean(word_length))
    upper_words = [x for x in real_review[i].split() if x.isupper()]
    real_uppers.append(len(upper_words))
    stopwords = [x for x in real_review[i].split() if x in Y]
    real_stopwords.append(len(stopwords))
    punc_words = [x for x in real_review[i] if x in string.punctuation]
    real_punc.append(len(punc_words))
    title_contain = real_review[i].lower().count(real_title[i].lower())
    real_contain_title.append(title_contain)



for i in range(0, len(fake_review)):
    word_length = [len(x) for x in fake_review[i].split()]
    fake_word_length.append(np.mean(word_length))
    upper_words = [x for x in fake_review[i].split() if x.isupper()]
    fake_uppers.append(len(upper_words))
    stopwords = [x for x in fake_review[i].split() if x in Y]
    fake_stopwords.append(len(stopwords))
    punc_words = [x for x in fake_review[i] if x in string.punctuation]
    fake_punc.append(len(punc_words))
    title_contain = fake_review[i].lower().count(fake_title[i].lower())
    fake_contain_title.append(title_contain)

print("Real Stopwords Count:", np.sum(real_stopwords))
print("Fake Stopwords Count:", np.sum(fake_stopwords))
print("Real Uppercase Words Count:", np.sum(real_uppers))
print("Fake Uppercase Words Count:", np.sum(fake_uppers))
print("Real Punctuations Count:", np.sum(real_punc))
print("Fake Punctuations Count:", np.sum(fake_punc))   
print("Real Word Length Average:", np.mean(real_word_length))
print("Fake Word Length Average:", np.mean(fake_word_length))
print("Real Review Length Average:", np.mean(real_review_length))
print("Fake Review Length Average:", np.mean(fake_review_length))
print("Product name appear count in Real Review:", np.sum(real_contain_title))
print("Product name appear count in Fake Review:", np.sum(fake_contain_title))
