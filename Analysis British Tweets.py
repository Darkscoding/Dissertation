# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:34:04 2020

@author: emigi
"""

import numpy as np
import pandas as pd

import os

os.chdir('C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part')

tweets_df = pd.read_csv('tweets_df.csv')

## Exporting other 1000 observations for human classification

fakenews_copy = pd.read_csv('observations')

labelling_df_2 = fakenews_copy.iloc[1001:2001, :]

labelling_df_2.to_excel('labelling_df_2.xlsx')

## exporing the dataset for human classification

tweets_df_export = tweets_df.copy()

tweets_df_export.to_excel('copy tweets df.xlsx')

## assessing the range of dates

min(tweets_df.date)

max(tweets_df.date)

## Data Pre Processing part: removing all rows with NAs from the dataset

tweets_df_noNan = tweets_df.dropna()

tweets_df_noNan.describe()

## approximately 70000 observations dropped

## Sentiment Analysis Part 1:
## Using pre-installed sentiment library, extracting polarity and subjectivity 
## out of each individual tweet

## for more information about polarization and how the algorithm works
## look at https://monkeylearn.com/sentiment-analysis/


from textblob import TextBlob

sentiments = []

for tweet in tweets_df_noNan.text:
    testimonial = TextBlob(tweet)
    sentiments.append(testimonial.sentiment.polarity)
    
tweets_df_noNan['Tweet Polarization Naive1 ML'] = sentiments

## Let us plot the histogram of such polarization
## We must consider that it is NOT over time here, so pretty much useless

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10))
plt.hist(tweets_df_noNan["Tweet Polarization Naive1 ML"])
plt.title('Histogram NOT over time of Tweets polarization')

## The classification clearly is not good at all
## I will need to use a supervised algorithm to increase the accuracy
