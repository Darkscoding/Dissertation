# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:20:09 2020

@author: emigi
"""

import numpy as np
import pandas as pd

import os

os.chdir('C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part')

british_tweets = pd.read_excel('copy tweets df.xlsx')

## Here (Thanks God) we have the tweets already classified. Only the need to classify them based on polarity

## Removing NAs from the adressee column

british_tweets = british_tweets.dropna(subset=['addressee'])
# british_tweets = british_tweets.iloc[:1355, :].dropna(subset=['Sentiment'])

## Still the need to use F1Score and not logloss here as well: imbalanced dataset

import matplotlib.pyplot as plt

import matplotlib.pyplot as pylab

## Setting the parameters of the next plots globally

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (70, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)

## Plotting the distribution of the humanly labelled tweets. N.B. Only 3 people here contributed to the labelling, conversely to the Italian articles

plt.hist(british_tweets.Sentiment.iloc[:1355], color='orange')
plt.title('Humanly Labelled British Tweets')

## Plotting the distribution of dates

# fig = plt.figure(figsize=(30, 10))
plt.hist(british_tweets.date, color='purple', bins=len(np.unique(british_tweets.date)))
plt.title('Distribution of British Tweets Dates')

date = british_tweets.date

import statistics
         
print("Less recent tweet in dataset")
print(min(date))

print("More recent tweet in dataset")
print(max(date))

print("Most frequent tweet in dataset")
print(statistics.mode(date))

## Analysis Part

## Introducing the Dummy to isolate the negative tweets

dummy_one = []

for sent in british_tweets.Sentiment:
    if sent == 1:
        dummy_one.append(1)
    else:
        dummy_one.append(0)
        
british_tweets['Dummy 1'] = dummy_one

## Testing Nltk with @ and hashtags

## Importing the code used for Petrologica Ltd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import linear_model
from tensorflow import keras
from sklearn.model_selection import GridSearchCV

seed = 43
#%matplotlib inline

## Making a copy of our dataframe

df = british_tweets.copy()

## Creating the Word Vectors

import nltk

test_tweet = british_tweets.text.iloc[200]

nltk.word_tokenize(test_tweet)

## It Works!


