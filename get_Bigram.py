# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:26:38 2017
@author: Neural
"""

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

def get_unigram_features(text_set):
    word_set = list()
    all_words = {}
    for line in text_set:
        word_unique = word_tokenize(line)
        word_set = word_set + word_unique
    for word in word_set:
        if word not in all_words:
            all_words[word] = 1
        else:
            all_words[word] += 1
    word_features = list(all_words)[:2000]
    return word_features

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

def get_extra_features(text_set):
	extra_features = list()
	for line in text_set:
		tmp = line[3:7]
		extra_features.append(tmp)
	return extra_features

def get_features_matrix(word_features,bigrams,train_review_text,extra_features):
    feartures_matrix = list()
    
    for j,item in enumerate(train_review_text):
        p = np.zeros([1,2504])
        word_unique = word_tokenize(item)
        for word in word_unique:
            if word in word_features:
                p[0][word_features.index(word)] = 1.0
        
        word_unique = word_tokenize(item)
        wf = list(nltk.bigrams(word_unique))
        for word in wf:
            if word in bigrams:
                p[0][2000+bigrams.index(word)] = 1.0

        p[0][2500] = extra_features[j][0]
        p[0][2501] = extra_features[j][1]
        p[0][2502] = extra_features[j][2]
        p[0][2503] = extra_features[j][3]
        
        feartures_matrix.append(p)
    return feartures_matrix  