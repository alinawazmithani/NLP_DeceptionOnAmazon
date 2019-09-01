# coding: utf-8
import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import nltk.stem
from nltk import bigrams

# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Label) = parseReview(line)
            rawData.append((Id, Text, Label))
            preprocessedData.append((Id, preProcess(Text), Label))


def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(preProcess(Text)),Label))
    for (_, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(preProcess(Text)),Label))


################
## QUESTION 1 ##
################

# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    return (int(reviewLine[0]), reviewLine[8], 'fake' if reviewLine[1] == '__label1__' else 'real')

# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
def preProcess(text):
    # Should return a list of tokens
    text = nltk.word_tokenize(text)
    #Bigrams
    text = list(bigrams(text))
    return text

################
## QUESTION 2 ##
################
featureDict = {} # A global dictionary of features


def toFeatureVector(words):
    featureVector = {}
    # return a dictionary 'featureVect' where the keys are the tokens in 'words' and the values are the number of occurrences of the tokens
    # start by using binary values only:
    for w in words:
        newW = '{}-{}'.format(w[0],w[1])
        featureVector[newW] = featureVector.get(newW,0) + 1
        featureDict[newW] = featureDict.get(newW,0) + 1
    return featureVector    

# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)

################
## QUESTION 3 ##
################

def crossValidate(dataset, folds):
    shuffle(dataset)
    accuracy = []
    precision = []
    recall = []
    f1score = []
    foldSize = int(len(dataset)/folds)
    for i in range(0,len(dataset),foldSize):
        #continue # Replace by code that trains and tests on the 10 folds of data in the dataset
        print("fold start %d foldSize %d" % (i,foldSize))
        myTestData = dataset[i:i+foldSize]
        myTrainData = dataset[:i] + dataset[i+foldSize:]
        classifier = trainClassifier(myTrainData)
        y_true = [x[1] for x in myTestData]
        y_pred = predictLabels(myTestData,classifier)
        accuracy.append(metrics.accuracy_score(y_true,y_pred,normalize=True))
        precision.append(metrics.precision_score(y_true,y_pred,average='weighted'))
        recall.append(metrics.recall_score(y_true,y_pred,average='weighted'))
        f1score.append(metrics.f1_score(y_true,y_pred,average='weighted'))
    accuracy_avg = np.mean(accuracy)
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1score_avg = np.mean(f1score)
    return accuracy_avg, precision_avg, recall_avg, f1score_avg

# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
#    return classifier.classify_many(map(lambda t: toFeatureVector(preProcess(t[0])), reviewSamples))
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# MAIN

# loading reviews
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'

## Do the actual stuff
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath)
# We split the raw dataset into a set of training data and a set of test data (80/20)
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

accuracy_result, precision_result, recall_result, f1score_result = crossValidate(trainData, 10)
print("Accuracy Result:")
print(accuracy_result)
print("Precision Result:")
print(precision_result)
print("Recall Result:")
print(recall_result)
print("F1Score Result:")
print(f1score_result)
