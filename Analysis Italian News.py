# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:23:48 2020

@author: emigi
"""

import numpy as np
import pandas as pd

import os

os.chdir('C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part')

italian_fakenews_df = pd.read_csv('observations')

import datetime

## Subselectig date and time from integer and inputting them into the dataset

date = []
time = []

for creation_date in italian_fakenews_df.created_time:
    convert_date = datetime.datetime.fromtimestamp(creation_date).date()
    convert_time = datetime.datetime.fromtimestamp(creation_date).time()
    date.append(convert_date)
    time.append(convert_time)
    
italian_fakenews_df['date'] = date
italian_fakenews_df['time'] = time

# labelling_df = italian_fakenews_df.iloc[:1000,:]

## Exporting the df to pre-label the dataframe

# labelling_df.to_excel("labelling_df.xlsx")

## importing the text

from newspaper import Article

articles_text = []
iterat = 0

for url in italian_fakenews_df.url:
    article = Article(url)
    
    try:
        article.download()
        article.parse()
        article.nlp()
    except:
        articles_text.append(0)
        pass
    
    articles_text.append(article.text)
    
    iterat += 1
    print(iterat)
    print(article.text)
    if iterat > 10000:
        print('10000 downloaded')
        condition = input('Input N if you want to continue, Y if you want to stop : ')
        if condition == 'Y':
            break
        else:
            pass
        
    if iterat > 20000:
        print('20000 downloaded')
        condition2 = input('Input N if you want to continue, Y to stop : ')
        if condition2 == 'Y':
            break
        else:
            pass
    
    if iterat > 30000:
        print('30000 downloaded')
        condition3 = input('Input N if you want to continue, Y to stop : ')
        if condition3 == 'Y':
            break
        else:
            pass
        
    elif iterat > 40000:
        print('40000 downloaded')
        condition4 = input('Input N if you want to continue, Y to stop : ')
        if condition4 == 'Y':
            break
        else:
            pass
        
    if iterat > 50000:
        print('50000 downloaded')
        condition5 = input('Input N if you want to continue, Y to stop : ')
        if condition5 == 'Y':
            break
        else:
            pass
        
    if iterat > 100000:
        
        print('100000 downloaded')
        condition6 = input('Input N if you want to continue, Y to stop : ')
        if condition6 == 'Y':
            break
        else:
            pass

## Exporting the dataset to an Excel file
            
# articles_text = pd.DataFrame(articles_text[:10001])
# articles_text.to_excel("10000ish Articles Text.xlsx")

articles_text = pd.read_excel("10000ish Articles Text.xlsx")
       
## Getting an idea of the distribution of dates
   
import statistics
         
print("Less recent article in dataset")
print(min(date))

print("More recent article in dataset")
print(max(date))

print("Most frequent date in dataset")
print(statistics.mode(date))

import matplotlib.pyplot as plt

import matplotlib.pyplot as pylab

## Setting the parameters of the next plots globally

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)

fig=plt.figure(figsize=(15, 10))
plt.hist(date, color='red', bins=7)
plt.title("Distribution of Dates, Italian Articles")

## I will conduct the iteration of train-test on 10000 observations first
## Importing the randomized human labels

test_classificationdf1 = italian_fakenews_df.iloc[:10001, :]

test_classificationdf1.to_excel("Fake News 10000.xlsx")

## Importing the first classification framework : 1700-ish articles classified
## Possible source of bias : I did not randomized the order of the articles showed
## to the humans. 

first_trial_df = pd.read_excel('Fake News 10000 1700 Classified.xlsx')

## Adding the text

first_trial_df['Article Text'] = articles_text.iloc[:10001, 1]

## Formatting data : removing the Nans and the 0s 

first_trial_df_labelled_noNans = first_trial_df.iloc[:1786].dropna()
first_trial_df_unlabelled = first_trial_df.iloc[1786:]

frames = [first_trial_df_labelled_noNans, first_trial_df_unlabelled]

first_trial_df_I_labelled = pd.concat(frames)

## Dropping rows where the Article Text = 0

first_trial_df_I_labelled = first_trial_df_I_labelled[first_trial_df_I_labelled['Article Text'] != 0]


## Beginning Naive Classification (Party Classification)
## Evaluation Metric used : F1-Score

## Why not using the logloss? 
## Imbalanced Dataset! Let us look at the distribution of parties for the
## already classified articles


## Plotting initial humanly classified articles to showcase imbalanced dataset

## Getting rid of human classification mistakes

classified_parties = first_trial_df_I_labelled.Party.iloc[:1780].dropna()

classified_parties = classified_parties[classified_parties != 'FL']
classified_parties = classified_parties[classified_parties != 'PI']

fig = plt.figure(figsize=(20, 15))
plt.hist(classified_parties, color='green', bins=5)
plt.title('Distribution of Humanly Classified Articles')



## First Iteration : O Vs FI, PD, M5, L

## Creation of the first dummy variables

dummy_O = []

for party in first_trial_df_I_labelled.Party:
    if party == 'O':
        dummy_O.append(1)
    elif party in ('FI', 'L', 'PD', 'M5'):
        dummy_O.append(0)
    else:
        dummy_O.append(np.nan)
        
first_trial_df_I_labelled['Dummy O'] = dummy_O

## Importing the code used for Petrologica Ltd

# import tensorflow as tf
from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import linear_model
# from tensorflow import keras
# from sklearn.model_selection import GridSearchCV

seed = 43
#%matplotlib inline

## Making a copy of our dataframe

df = first_trial_df_I_labelled.copy()

## Creating the Word Vectors

import nltk

## TO DO: Adapt Nltk to Italian Words and Sentences
                                                                                                                                                                                           
def dictionary_words(X):                                          ## In this function, for each article we associate a list of numbers representing the categorical variables related to tokens
    features_pre_classified_list = X.tolist()
    
    sentences_pre_classified = []
    for article in features_pre_classified_list:
        if isinstance(article, int):
            pass
        else:
            #superstring = superstring.join(sentence)  ##Memory Error in Python
            try:
                tokens = nltk.word_tokenize(article)
                sentences_pre_classified.append([tokens])
            except:
                sentences_pre_classified.append(0)
    
    
    # print(sentences_pre_classified)
    
    mydict={}
    count = 0
    i = 0
    for sent in sentences_pre_classified:
        count += 1
        if sent != 0:
            for token in sent[0]:
                # print(count)
                # print(token)
                if(i>0 and token in mydict):
                    continue
                else:    
                    i = i+1
                    mydict[token] = i
    
    
    count = 0
    k_pre_classified = [[] for _ in range(len(sentences_pre_classified))]
    for sent in sentences_pre_classified:
        
        if sent != 0:
            for token in sent[0]:
                k_pre_classified[count].append(mydict[token])
        else:
            k_pre_classified[count].append(0)
        
        count += 1
    
    return k_pre_classified

## Formatting dict to create a matrix
## introducing 0 randomly in the list and permuting randomly with numpy

def formattingX(X):
    assert isinstance(X, list), 'X must be a list'
    
    bestl = len(X[0])
    
    # print(bestl)
    for el in X[1:]:
        l = len(el)
        if l > bestl:
            bestl = l
    # print(bestl)
            
    ##This if-statement ensures that if there isa chunk of articles with len(dictionary)
    ##then it is adapted to fit 3533
    
    if bestl < 3533:
        index = 0
        for el2 in X:
            l2 = len(el2)
            diff = bestl - l2
            mode_l = [int(np.mean(el2))]
            if diff != 0:
                el2 = el2 + mode_l*diff + mode_l*(3533 - (l2 + diff))
            else:
                el2 = el2 + mode_l*(3533 - l2)
            el2 = np.random.permutation(el2).tolist()
            el2 = np.random.permutation(el2).tolist()
            el2 = np.random.permutation(el2).tolist()
            X[index] = el2
            index += 1
    
    else:
        index = 0        
        for el2 in X:
            l2 = len(el2)
            mode_l = [int(np.mean(el2))]
            if l2 == bestl:
                # print('l2 == bestl')
                # print(l2)
                pass
            elif l2 < bestl:
                el2 = el2 + mode_l*(bestl - l2)
                el2 = np.random.permutation(el2).tolist()
                el2 = np.random.permutation(el2).tolist()
                el2 = np.random.permutation(el2).tolist()  
                if bestl > 4000:
                    X[index] = np.random.sample(el2, 3533)
                else:
                    X[index] = el2
            # print('l2 < bestl')
            # print(len(el2))
            index += 1
    
    ## Apparently with lengths greater than 4000 the clfbest.fit has some problems
    ## solution: randomly sample from each list
    
    return X

## N.B.
## We start the iteration with the classified part and then proceed to cover the whole dataset

class dynamic_stoc_gradient_models_withCV_base():
    def __init__(self, df, division_percentage, verbose=0, verboselog=0, verbosesvm=0, comparison="f1score", seed=None):
        if isinstance(seed, type(None)):
            seed = np.random.seed()
        self.seed = seed
        self.verbose = verbose
        self.division_percentage = division_percentage
        self.verboselog = verboselog
        self.verbosesvm = verbosesvm
        self.df = df
        self.comparison = comparison
        self.df = df
        subsetDataFrame_relevant = df[df['Dummy O'] == 1]
        subsetDataFrame_irrelevant = df[df['Dummy O'] == 0]
        frames = [subsetDataFrame_irrelevant, subsetDataFrame_relevant]
        starting_train = pd.concat(frames)
        starting_train = starting_train.sample(frac=1)
        starting_train = starting_train.sample(frac=1)
        starting_train = starting_train.sample(frac=1)
        self.starting_train = starting_train
        n = starting_train.shape[0]
        self.n = n
        max_iter = np.ceil(10**6 / n)
        self.max_iter = max_iter
        
        X = np.array(formattingX(dictionary_words(starting_train['Article Text'])))
        initarrayX = X[0]
        index = 1
        for l in X[1:]:
            if index == 1:
                arrayX = np.vstack((initarrayX, l))
            else:
                arrayX = np.vstack((initarrayX, arrayX))
            index += 1
        self.X = arrayX
        nmax = len(X)
        y = np.array(starting_train['Dummy O'].iloc[:nmax], dtype=int)
        self.y = y
    
    def train_test_division(self):
        df = self.df
        n = self.n
        division_percentage = self.division_percentage
        n_test = int((n / division_percentage[0])*division_percentage[1])
        train_data = df.iloc[:n, :]
        test_data = df.iloc[n:(n+n_test)+1, :]
        return (train_data, test_data)
    
    def alpha_regularization_CV_logit(self):
        seed = self.seed
        verbose = self.verbose
        verboselog = self.verboselog
        
        alpha = 10.0**-np.arange(1,7)
        X = self.X
        y = self.y
        
        # print(X.shape)
        # print(y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        
        # print(X_train.shape)
        # print(y_train.shape)
        
        clfbest = linear_model.SGDClassifier(random_state=seed, loss='log', penalty='elasticnet', verbose=verboselog, alpha=10.0**-1, n_iter_no_change=10, n_jobs=-1, class_weight='balanced')
        try:
            clfbest.fit(X_train, y_train)
        except:
            clfbest.fit(X_train.reshape(-1, 1), y_train)
        bestf1score = f1_score(y_test, clfbest.predict(X_test))
        for a in alpha:
            clf = linear_model.SGDClassifier(random_state=seed, loss='log', penalty='elasticnet', verbose=verboselog, alpha=a, learning_rate='optimal', n_iter_no_change=7, n_jobs=-1, class_weight='balanced')    
            clf.fit(X_train,y_train)
            f1score = f1_score(y_test, clf.predict(X_test))
            if verbose == 2:
                print(f"f1score: {f1score}")
            if f1score > bestf1score:
                bestf1score = f1score
                clf = clfbest
        
        besttscore = clfbest.score(X_train, y_train)
        bestvscore = clfbest.score(X_test, y_test)
        
        if verbose == 1:
            print('Confusion Matrix')
            print(f'{metrics.confusion_matrix(clfbest.predict(X_test), y_test)}')
            print('Best f1score')
            print(f'{bestf1score}')
            print('Best Train Score')
            print(f'{besttscore}')
            print('Best test score')
            print(f'{bestvscore}')
        return (clf, besttscore, bestvscore, bestf1score)
    
    def alpha_regularization_CV_SVM(self):
        seed = self.seed
        verbose = self.verbose
        verbosesvm = self.verbosesvm
        alpha = 10.0**-np.arange(1,7)
        X = self.X
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        
        clfbest = linear_model.SGDClassifier(random_state=seed, verbose=verbosesvm, alpha=10.0**-1, n_iter_no_change=10, n_jobs=-1, class_weight='balanced')
        
        clfbest.fit(X_train, y_train)
        bestf1score = f1_score(y_test, clfbest.predict(X_test))
        for a in alpha:
            clf = linear_model.SGDClassifier(random_state=seed, verbose=verbosesvm, alpha=a, learning_rate='optimal', n_iter_no_change=7, n_jobs=-1, class_weight='balanced')    
            clf.fit(X_train ,y_train)
            f1score = f1_score(y_test, clf.predict(X_test))
            if verbose == 2:
                print(f"f1score: {f1score}")
            if f1score > bestf1score:
                bestf1score = f1score
                clf = clfbest
        
        besttscore = clfbest.score(X_train, y_train)
        bestvscore = clfbest.score(X_test, y_test)
        
        if verbose == 1:
            print('Confusion Matrix')
            print(f'{metrics.confusion_matrix(clfbest.predict(X_test), y_test)}')
            print('Best f1score')
            print(f'{bestf1score}')
            print('Best Train Score')
            print(f'{besttscore}')
            print('Best test score')
            print(f'{bestvscore}')
        
        return (clf, besttscore, bestvscore, bestf1score)
        
    def whence(self):
        clflog, tscorelog, vscorelog, f1scorelog = self.alpha_regularization_CV_logit()
        clfsvm, tscoresvm, vscoresvm, f1scoresvm = self.alpha_regularization_CV_SVM()
        if self.comparison == "f1score":
            if f1scorelog > f1scoresvm:
                return ([1, 0], clflog, tscorelog, vscorelog, f1scorelog, tscoresvm, vscoresvm)
            elif f1scorelog < f1scoresvm:
                return ([0, 1], clfsvm, tscorelog, vscorelog, f1scoresvm, tscoresvm, vscoresvm)
            elif f1scorelog == f1scoresvm:
                return ([1,1], clflog, clfsvm, tscorelog, vscorelog, f1scorelog, tscoresvm, vscoresvm, f1scoresvm)
    
    ## TO DO : apply dictionary_words to test['Article Text'], label the test part
    ## and concatenate train and test
    
    def classification(self):
        whence = self.whence()
        train, test = self.train_test_division()
        verbose = self.verbose
        df = self.df
        
        ## dropping possible labelled rows in test
        
        indexes_dummyone = test[test['Dummy O'] == 1].index.tolist()
        indexes_dummyzero =  test[test['Dummy O'] == 0].index.tolist()
        
        if verbose == 2:
            print('Number of dropped indexes in test')
            print(len(indexes_dummyone) + len(indexes_dummyzero))
            print('Dropped Indexes in test')
            print(indexes_dummyone)
            print(indexes_dummyzero)
            
        
        test = test.drop(indexes_dummyone)
        test = test.drop(indexes_dummyzero)
        
        X_test = np.array(formattingX(dictionary_words(test['Article Text'])))
        
        initarrayX = X_test[0]
        index = 1
        print(len(X_test[1]))
        for l in X_test[1:]:
            if index == 1:
                arrayX = np.vstack((initarrayX, l))
            else:
                arrayX = np.vstack((initarrayX, arrayX))
            index += 1
        
        X_test = arrayX
        
        if whence[0] == [1, 0]:
            clf_class = whence[1]
            predictions = clf_class.predict(X_test)
        elif whence[0] == [0, 1]:
            clf_class = whence[1]
            predictions = clf_class.predict(X_test)
        elif whence[0] == [1,1]:
            clf_class = whence[np.random.randint(1, 3)] ##it choices randomly svm or logistic regression
            predictions = clf_class.predict(X_test)
            
        test['Dummy O'] = predictions
        last_index = test.index[-1]
        frames = [train, test, df.iloc[last_index+1:, :]]
        final_df = pd.concat(frames)
        
        return final_df

class dynamic_stoc_gradient_models_withoutCV_base():
    def __init__(self, df, division_percentage, verbose = 0, seed=None, comparison = "f1score"):
        if isinstance(seed, type(None)):
            seed = np.random.seed()
        if not isinstance(division_percentage, tuple):
            print('Warning: use a tuple to specify train/test division')
        self.seed = seed
        self.comparison = comparison
        df = df.sample(frac=1)
        self.df = df
        subsetDataFrame_relevant = df[df['Relevant - production guidance'] == 1.0]
        subsetDataFrame_irrelevant = df[df['Relevant - production guidance'] == 0.0]
        starting_train = subsetDataFrame_relevant.append(subsetDataFrame_irrelevant)
        starting_train = starting_train.sample(frac=1)
        starting_train = starting_train.sample(frac=1)
        starting_train = starting_train.sample(frac=1)
        self.starting_train = starting_train
        n = starting_train.shape[0]
        self.n = n
        max_iter = np.ceil(10**6 / n)
        self.max_iter = max_iter
        self.division_percentage = division_percentage
        self.verbose = verbose
    
    def train_test_division(self):
        df = self.df
        n = self.n
        division_percentage = self.division_percentage
        n_test = int((n / division_percentage[0])*division_percentage[1])
        train_data = df.iloc[:n+1, 0]
        test_data = df.iloc[n:(n+n_test)+1,0]
        return (train_data, test_data)
    
    ## Fix Issue here
    def LogisticRegressionModel(self):
        df = self.starting_train
        verbose = self.verbose
        X = dictionary_words(df['Text'])
        nmax = len(X)
        y = np.array(df['Relevant - production guidance'][:nmax])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        clf = LogisticRegression(C=0.5)
        clf.fit(X_train, y_train)
        tscore = clf.score(X_train, y_train)
        vscore = clf.score(X_test, y_test)
        if max(clf.predict(X_test)) == 1.0:
            f1score = f1_score(y_test, clf.predict(X_test))
        else:
            f1score = f1_score(y_test, clf.predict(X_test), average='micro')
        if verbose != 0:
            print(f"tscore={tscore} vscore={vscore}")
            print("Confusion Matrix")
            print(f"{confusion_matrix(clf.predict(X_test), y_test)}")
            print("F1 Score")
            print(f"{f1score}")
        return tscore, vscore, f1score
        
    def SVMModel(self):
        df = self.starting_train
        seed = self.seed
        verbose = self.verbose
        X = dictionary_words(df['Text'])
        nmax = len(X)
        y = np.array(df['Relevant - production guidance'][:nmax])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        clf = SVC(random_state=seed, verbose=4) ## verbose variable for printing
        clf.fit(X_train, y_train)
        tscore = clf.score(X_train, y_train)
        vscore = clf.score(X_test, y_test)
        f1score = f1_score(y_test, clf.predict(X_test))
        if verbose != 0:
            print(f"tscore = {tscore} vscore = {vscore}")
            print("Confusion Matrix")
            print(f"{confusion_matrix(clf.predict(X_test), y_test)}")
            print("F1 Score")
            print(f"{f1score}")
        return tscore, vscore, f1score
    
    def whence(self):
        tscorelog, vscorelog, f1scorelog = self.LogisticRegressionModel()
        tscoresvm, vscoresvm, f1scoresvm = self.SVMModel()
        if self.comparison == "f1score":
            if f1scorelog > f1scoresvm:
                return [1, 0]
            elif f1scorelog < f1scoresvm:
                return [0, 1]
            elif f1scorelog == f1scoresvm:
                return [1,1]
            

## Testing whence with logistic regression and linear SVM with Stochastic Gradient Descent
## Different alphas (learning rates) are tested in each iteration, trying to choose the best
## Possible Problems : the hinge loss (a.k.a. SVM) in the Stochastic Gradient Descent Function
## features only a Soft Margin (C small, e.g. C = 10). It is also a LINEAR SVM, thus
## we might improve on it with non-linear projections and non-linear kernels
                
TestD1 = dynamic_stoc_gradient_models_withCV_base(df, (0.7, 0.3), verbose=1)
test_whence = TestD1.whence()

train_df_D1 = TestD1.train_test_division()[0]
test_df_D1 = TestD1.train_test_division()[1]
test_finaldf = TestD1.classification()



## First iteration gives a best f1score of 70%. Quite good! HOWEVER, the CONFUSION MATRIX features HALF CORRECTLY CLASSIFIED compared to INCORRECTLY CLASSIFIED.
## We will observe how the f1score and the confusion matrix updates for each iteration
## We will also compare test and train losses to account for possible underfit or overfit problems in the iterations

## Iterations over the 8500 dataset

n = TestD1.n

df_i = df.copy()

df_i = df_i.dropna(subset=['Article Text'])

scores = []
f1scores = []
whence = []

n_max = 8000
iterat = 0


while n <= n_max:
    Di = dynamic_stoc_gradient_models_withCV_base(df_i, (0.7, 0.3), verbose=1)
    
    ## storing whence
    
    whence_i =list(Di.whence())
    
    # print(whence_i)
    
    whence.append(whence_i[0])
    
    ## storing scores
    
    # print(whence[-1])
    
    if whence[-1] == [1, 1]:
        scores.append([whence_i[3],  whence_i[4], whence_i[6], whence_i[7]])
        f1scores.append(whence_i[5])    
    else:
        scores.append([whence_i[2],  whence_i[3], whence_i[5], whence_i[6]])
        f1scores.append(whence_i[4])
        
    ## classification and substitution
        
    # print(Di.train_test_division())
      
    df_i = Di.classification()
    n_subsetDataFrame_relevant = df_i[df_i['Dummy O'] == 1].shape[0]
    n_subsetDataFrame_irrelevant = df_i[df_i['Dummy O'] == 0].shape[0]
    n_classified = n_subsetDataFrame_relevant + n_subsetDataFrame_irrelevant
    print(f'Articles Classified : {n_classified}')
    # except:
    #     print('Ended before completion')
    #     n_subsetDataFrame_relevant = df_i[df_i['Dummy O'] == 1].shape[0]
    #     n_subsetDataFrame_irrelevant = df_i[df_i['Dummy O'] == 0].shape[0]
    #     n_classified = n_subsetDataFrame_relevant + n_subsetDataFrame_irrelevant
    #     print(f'Articles Classified : {n_classified}')
    #     break
    print(f'Iteration : {iterat}')
    
    ## Optional
    
    
    # print(scores[-1])
    # print(f1scores[-1])
    
    iterat += 1
    
## plotting the f1scores at each iteration, I Trial (8000ish observations)
   
iterat = list(range(1, 4))    

logistic_regression_scores_train = scores[:][0]
logistic_regression_scores_test = scores[:][1]
svm_train = scores[:][2]
svm_test = scores[:][3]

data = ((iterat, f1scores), (iterat, logistic_regression_scores_train), (iterat, logistic_regression_scores_test), 
        (iterat, svm_train), (iterat, svm_test))
colors = ('blue', 'red',
          'green', 'black')
groups = ('f1score', 'Logit Train Acc.', 'Logit Test Acc.', 'SVM Train Acc.', 'SVM Test Acc.')

fig= plt.figure(figsize=(30,15))
ax = fig.add_subplot(1, 1, 1)

for dat, color, group in zip(data, colors, groups):
    x, y = dat
    ax.plot(x, y, c=color, label=group)

plt.legend(loc=2)
plt.xlabel('Iteration')
plt.ylabel('Accuracy and F1Score')
plt.title('Result First Dynamic Test (8000ish observations)')
plt.show()

## The Results seem promising as the f1score did not drop never below 0.65. However, it decreases overtime

## Now, let's try with a random sample of the original dataset

sample_one_percent = italian_fakenews_df.iloc[15000:, :].sample(frac=0.01)

# sample_one_percent.to_excel("Sample One Percent Italian Fake News df.xlsx")

## By comparing the histograms of the plots, the distribution of the date of the articles is similar.

## By opening different kernels (10), I will try to speed up the process of data scraping

## Let's download as many articles text as possible

import numpy as np

import pandas as pd

import os

from newspaper import Article

import os

os.chdir('C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part')

sample_one_percent = pd.read_excel("Sample One Percent Italian Fake News df.xlsx")

n = sample_one_percent.shape[0]

nI = int(n/10)
nII = nI*2
nIII = nI*3
nIV = nI*4
nV = nI*5
nVI = nI*6
nVII = nI*7
nVIII = nI*8
nIX = nI*9

articles_text = []
iterat = 0

for url in sample_one_percent.iloc[nIX:n, :].url:
    article = Article(url)
    
    try:
        article.download()
        article.parse()
        article.nlp()
    except:
        articles_text.append(0)
        pass
    
    articles_text.append(article.text)
    
    iterat += 1
    print(iterat)
    
## Saving the articles in 10 files
    
articles_df = pd.DataFrame(articles_text)

## Loading the articles and combine them together

articlesI = pd.read_excel('Articles Text I.xlsx')
articlesII = pd.read_excel('Articles Text II.xlsx')
articlesIII = pd.read_excel('Articles Text III.xlsx')
articlesIV = pd.read_excel('Articles Text IV.xlsx')
articlesV = pd.read_excel('Articles Text V.xlsx')
articlesVI = pd.read_excel('Articles Text VI.xlsx')
articlesVII = pd.read_excel('Articles Text VII.xlsx')
articlesVIII = pd.read_excel('Articles Text VIII.xlsx')
articlesIX = pd.read_excel('Articles Text IX.xlsx')
articlesX = pd.read_excel('Articles Text X.xlsx')

frames = [articlesI, articlesII, articlesIII, articlesIV, articlesV, 
          articlesVI, articlesVII, articlesVIII, articlesIX, articlesX]

articles_series = pd.concat(frames)

sample_one_percent['Article Text'] =list(articles_series.iloc[:68036, 1])

## adding the already classified 

sample_one_percent['Dummy O'].iloc[:5]

frames = [df_i, sample_one_percent]

df_final = pd.concat(frames)

## removing Nans from the article text column

df_final = df_final.dropna(subset=['Article Text'])

## Exporting df_final to look at the result

df_final.to_excel('Final Dataframe Italian Articles.xlsx')

## starting the algorithm ; plot f1scores, train and test scores over iterations

n = df_i.shape[0]

df_i_final = df_final.copy()

scores = []
f1scores = []
whence = []

n_max = 8000
iterat = 0



while n <= n_max:
    Di = dynamic_stoc_gradient_models_withCV_base(df_i_final, (0.7, 0.3), verbose=0)
    
    ## storing whence
    
    whence_i =list( Di.whence())
    
    # print(whence_i)
    
    whence.append(whence_i[0])
    
    ## storing scores
    
    # print(whence[-1])
    
    if whence[-1] == [1, 1]:
        scores.append([whence_i[3],  whence_i[4], whence_i[6], whence_i[7]])
        f1scores.append(whence_i[5])    
    else:
        scores.append([whence_i[2],  whence_i[3], whence_i[5], whence_i[6]])
        f1scores.append(whence_i[4])
        
    ## classification and substitution
        
    # print(Di.train_test_division())
    
    try:    
        df_i_final = Di.classification()
        n_subsetDataFrame_relevant = df_i_final[df['Dummy O'] == 1].shape[0]
        n_subsetDataFrame_irrelevant = df_i_final[df_i_final['Dummy O'] == 0].shape[0]
        n_classified = n_subsetDataFrame_relevant + n_subsetDataFrame_irrelevant
        print(f'Articles Classified : {n_classified}')
    except:
        n_subsetDataFrame_relevant = df_i_final[df['Dummy O'] == 1].shape[0]
        n_subsetDataFrame_irrelevant = df_i_final[df_i_final['Dummy O'] == 0].shape[0]
        n_classified = n_subsetDataFrame_relevant + n_subsetDataFrame_irrelevant
        print('Ended before completion')
        print(f'Articles Classified : {n_classified}')
        break
    print(f'iteration : {iterat}')
    
    ## Optional
    
    
    # print(scores[-1])
    print(f'f1score : {f1scores[-1]}')
    
    iterat += 1
    
## Export the final dataset in Excel and look at the result
    
df_i_final.to_excel('Test 2 Final Classification Dataset.xlsx')

## Testing the singular functions to debug

TestD2 = dynamic_stoc_gradient_models_withCV_base(df_final, (0.7, 0.3), verbose=1)

test_whence = TestD2.whence()
test_df_D2 = TestD2.train_test_division()[1]
test_finaldf = TestD2.classification()

X_test_2 = formattingX(dictionary_words(test_df_D2['Article Text']))
X_test_1 = formattingX(dictionary_words(test_df_D1['Article Text']))
X_train_1 = formattingX(dictionary_words(df_i['Article Text']))

## Same dimension

## TO DO: solve the bug with whence

## Naive appending of articles to a list by looking at specific keywords for each party

party = []

for text in sample_one_percent['Article Text']:
    if not isinstance(text, str):
        party.append(np.nan)
    else:
        L_count = text.count('Lega') + text.count('lega')
        M5_count = text.count('Movimento 5 stelle') + text.count('Movimento 5 Stelle') + text.count('movimento 5 stelle')
        PD_count = text.count('Partito Democratico') + text.count('partito democratico')
        FI_count = text.count('Forza Italia') + text.count('forza italia')
        if L_count > M5_count + PD_count + FI_count:
            party.append('L')
        elif M5_count > L_count + PD_count + FI_count:
            party.append('M5')
        elif FI_count > L_count + PD_count + M5_count:
            party.append('FI')
        elif PD_count > M5_count + L_count + FI_count:
            party.append('PD')
        else:
            party.append('O')
        
sample_one_percent['Party'] = party
        
## Removing the rows where party == nan

sample_one_percent = sample_one_percent.dropna(subset=['Party'])

## 20000 removed. If needs be, I will replace them with the mode
## Exporting the dataset to be analyzed in the advanced section

sample_one_percent.to_excel('Sample One Percent Italian Fakenews Party Labelled Naively.xlsx')
