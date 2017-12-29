import nltk
import sqlite3
import pandas as pd
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier

conn = sqlite3.connect('./yelpHotelData/yelpHotelData.db')
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "Y"'
fake = pd.read_sql(query, conn)
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "N"'
real = pd.read_sql(query, conn)
conn.close()

fake = fake.sample(750)
fake['tag'] = 'fake'
real = real.sample(750)
real['tag'] = 'real'
df = pd.concat([fake,real])
df = df.iloc[np.random.permutation(len(df))]

df['reviewContent'] = df.apply(lambda row: nltk.word_tokenize(row['reviewContent']), axis=1)
all_words = nltk.FreqDist(word.lower() for row in df['reviewContent'] for word in row)
word_features = list(all_words)[:2000]


df['bigrams'] = df.apply(lambda row: list(nltk.bigrams(row['reviewContent'])), axis=1)
bigrams = nltk.FreqDist(word[0].lower() +" "+ word[1].lower() for row in df['bigrams'] for word in row)
bigrams = list(bigrams)[:500]




import nltk
import numpy as np
from nltk.tokenize import word_tokenize

def get_bigram_features(text_set):
    word_set = list()
    bigrams = list()
    all_bigrams_words = {}
    for line in text_set:
        word_unique = word_tokenize(line)
        wf = list(nltk.bigrams(word_unique))
        word_set = word_set + wf

    for word in word_set:
        if word not in all_bigrams_words:
            all_bigrams_words[word] = 1
        else:
            all_bigrams_words[word] += 1
    bigrams = list(all_bigrams_words)[:500]
    return bigrams


text_set = ['along with its tag','Returns the features of a reviewReturns']

wf = nltk.bigrams(text_set)
w = list(wf)






del(all_words)
del(real)
del(fake)

def document_features(doc):
    """Returns the features of a review, along with its tag"""
    
    document_words = set(doc['reviewContent'])
    features = {}
    # Grabbing the bigrams
    
    
    bigSet = []
    for word in doc['bigrams']:
        bigSet.append(word[0].lower() + " " +word[1].lower())
    bigSet = set(bigSet)
    for word in bigrams:
        features['bigram: ' + word] = (word in bigSet)

    # Counting the pronoun usage
    meCount = 0
    for word in doc['reviewContent']:
        if (word.lower() == 'i' or word.lower() == 'me'):
            meCount += 1
    me = False
    if (meCount > 5):
        me = True

    for word in word_features:
        features['contains ' + word] = (word in document_words)
    features.update(
    {'rating': doc['rating'], 'useful': doc['usefulCount'], 'cool': doc['coolCount'], 'funny': doc['funnyCount'], 'meCount': me})
    return [features,doc['tag']]

t=list(test_set)

featuresets = df.apply(document_features, axis = 1)
train_set, test_set = featuresets[250:], featuresets[:250]
del(featuresets)


t = list(test_set) 
p = list(train_set)










classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(20)

from sklearn.svm import SVC
svmClass = SklearnClassifier(SVC()).train(train_set)
print("SVM Classifier:")
print(nltk.classify.accuracy(svmClass, test_set))


from sklearn.ensemble import AdaBoostClassifier
adaClass = SklearnClassifier(AdaBoostClassifier()).train(train_set)
print("Adaboost Classifier:")
print(nltk.classify.accuracy(adaClass, test_set))


from sklearn.neural_network import MLPClassifier
nnClass = SklearnClassifier(MLPClassifier()).train(train_set)
print("Neural Network Classifier:")
print(nltk.classify.accuracy(nnClass, test_set))


def svm_features(doc):
    document_words = set(doc['reviewContent'])
    features = {}
    # Grabbing the bigrams
    bigSet = []
    for word in doc['bigrams']:
        bigSet.append(word[0].lower() + " " +word[1].lower())
    bigSet = set(bigSet)
    for word in bigrams:
        features['bigram: ' + word] = (word in bigSet)

    # Counting the pronoun usage
    meCount = 0
    for word in doc['reviewContent']:
        if (word.lower() == 'i' or word.lower() == 'me'):
            meCount += 1
    me = False
    if (meCount > 5):
        me = True

    for word in word_features:
        features['contains ' + word] = (word in document_words)
    features.update(
    {'rating': doc['rating'], 'useful': doc['usefulCount'], 'cool': doc['coolCount'], 'funny': doc['funnyCount'], 'meCount': me})
    return features

    
    
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear','poly', 'rbf'), 'C':[1, 10]}
vec = DictVectorizer()
trainTags, testTags = df[250:],df[:250]
svmSet = df.apply(svm_features, axis = 1)
svmSet = vec.fit_transform(svmSet).toarray()
svmTest, svmTrain = svmSet[:250],svmSet[250:]




svr = SVC()
svmClass = GridSearchCV(svr, parameters)
svmClass.fit(svmTrain,trainTags['tag'])
print("SVM with Grid Search Cross-Validation: ")
print(svmClass.score(svmTest,testTags['tag']))
