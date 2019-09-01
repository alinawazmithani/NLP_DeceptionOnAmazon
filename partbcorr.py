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


rawData = []

def loadData(path, Text=None):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label, Rating, Verified, Category) = parseReview(line)
            rawData.append((Id, Text, Label, Rating, Verified, Category))
            
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    return (int(reviewLine[0]), reviewLine[8], 'fake' if reviewLine[1] == '__label1__' else 'real', reviewLine[2], reviewLine[3], reviewLine[4])

loadData('amazon_reviews.txt')
real_count = []
fake_count = []
for i in range(0, len(rawData)):
    if(rawData[i][2] == 'real'):
        real_count.append(rawData[i][4])
    else:
        fake_count.append(rawData[i][4])
#plt.hist(real_count)
#plt.hist(fake_count)
df = pd.DataFrame(rawData, columns = ['ID', 'Text', 'Label', 'Rating', 'Verified', 'Category'])
df.apply(lambda x: x.factorize()[0]).corr()
