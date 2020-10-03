# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:34:22 2020

@author: emigi
"""

import numpy as np
import pandas as pd

import os
os.chdir('C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part')

first_trial_df = pd.read_excel('Fake News 10000 1700 Classified.xlsx')

articles_text = pd.read_excel("10000ish Articles Text.xlsx")

first_trial_df['Article Text'] = articles_text.iloc[:, 1]

## concatenating the classified articles with the party labelled one

sample_one_percent_labelled = pd.read_excel('Sample One Percent Italian Fakenews Party Labelled Naively.xlsx')

## Labelling the parties in the first_trial_df as well

party = []

for text in first_trial_df['Article Text'].iloc[1786:]:
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
            
first_trial_df.Party.iloc[1786:] = party

## Filtering out the nan

first_trial_df = first_trial_df.dropna(subset=['Party'])

## Creating a whole dataset

frames = [first_trial_df, sample_one_percent_labelled]

final_df = pd.concat(frames)

from tqdm import tqdm
import re
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import nltk

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, merge, Masking,TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

## Testing first iteration

glove_path = 'C:/Users/emigi/OneDrive/Desktop/Dissertation/Analysis Files/NLP Part/glove_WIKI'

glove_model = Word2Vec.load('glove_WIKI')



## Loading the italian word embeddings

## Create a dictionary covering all possible words in the sample

starting_train = final_df.dropna(subset=['Sentiment'])

starting_train = starting_train[starting_train['Article Text'] != 0]
starting_train = starting_train.dropna(subset=['Article Text'])
final_df = final_df[final_df['Article Text'] != 0]
final_df = final_df.dropna(subset=['Article Text'])

## Here is the approach: 
## let us get the article with the most number of tokens and look at it
## Then: create the word vector of the most n frequent words
## compute the of such vector and stack it up with the others to create an
## array

lengths_tokens = []
iterat = 0

for art in final_df['Article Text']:
    unique_tokens = []
    tokens = nltk.word_tokenize(art, language='italian')
    for t in tokens:
        unique_tokens.append(t)
    l = len(unique_tokens)
    lengths_tokens.append(l)
    print(iterat)
    iterat += 1

# The longest tokenized article has length : 6709  
# The smallest has length : 4
# Deletion of useless articles
    
index_l = final_df.index

iterat = 0
for (l, index) in zip(lengths_tokens, index_l):
    if l < 100:
        try:
            final_df = final_df.drop(index)
        except:
            pass
    else:
        pass
    print(iterat)
    iterat += 1
    
    
## Now, for each article, let us create a dictionary with the tokens
## such that it will be possible to fetch the most frequent ones
    
dict_list = []
iterat = 0

punct_mapping = ("‘",",", "'", "₹", "$", "´", "'", "°", "degrees", "€", "$", "™", "tm", "√", " sqrt ", "×", "x", "²", "2", "—", "-", "–", "-", "’", "'", "_", "-", "`", "'", '“', '"', '”', '"', '“', '"', "£", "$", '∞', 'infinity', 'θ', 'theta', '÷', '/', 'α', 'alpha', '•', '.', 'à', 'a', '−', '-', 'β', 'beta', '∅', 'empty', '³', '3', 'π', 'pi')
articoli_and_others = ('il', 'la', 'gli', 'le', 'loro', 'essi', 'egli', 'che', 'con', 'una', 'un', 'a', 'tra', 'in', 'con', 'su', 'per', 'tra', 'fra')

for art in final_df['Article Text']:
    d = {}
    tokens = nltk.word_tokenize(art, language='italian')
    for t in tokens:
        if t in punct_mapping or t in articoli_and_others:
            pass
        else:
            if t in d:
                d[t] += 1
            else:
                d[t] = 1
    dict_list.append(d)
    print(iterat)
    iterat += 1

final_df['Dictionary Words'] = dict_list

final_df.to_excel('Final Df Advanced Italian News.xlsx')




## Defining initial train and test

final_df = pd.read_excel('Final Df Advanced Italian News.xlsx')

def train_test_division(starting_train, df, division_percentage):
    n = starting_train.shape[0]
    n_test = int((n / division_percentage[0])*division_percentage[1])
    test_data = df.iloc[n:(n+n_test)+1, :]
    return (starting_train, test_data)

starting_train, starting_test = train_test_division(starting_train, final_df, (0.7, 0.3))

class Attention(Layer):                                              
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

from keras import losses
from keras import metrics
  
def model_bilstm_attention(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    
    l = Bidirectional(LSTM(64, return_sequences=True, go_backwards=True))(x)
    g = Bidirectional(GRU(64, return_sequences=True))(x)
    x = concatenate([l, g])
    x = Bidirectional(LSTM(64//2, return_sequences=True))(x)
    
    
    # x1 = Attention(maxlen)(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = GlobalMaxPooling1D()(x)
    
    x = Concatenate()([x2, x3])
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def formatting_matrix_X(starting_train_df):
    
    
    ## Get the top 50 words in order of presence and get their vector
    ## Then, compute the mean of each
    word_embedding_X = np.ones(10)
    itera = 0
    for d in starting_train_df['Dictionary Words']:
        means_array = []
        sort_dict = sorted(d, key=d.get, reverse=True)
        top_10 = sort_dict[:10]      ## 10 values for each article
        for word in top_10: 
            try:
                word_vector = glove_model.wv[word]
                mean = np.mean(word_vector)
                means_array.append(mean)
                itera += 1
            except:
                pass  
        print(itera)                                
        
        if len(means_array) < 10:
            print(f'Length of means_array : {len(means_array)}')
            means_array = means_array + [np.mean(means_array)]*(10 - len(means_array))
           
        
        means_array = np.array(means_array)
        word_embedding_X = np.vstack((word_embedding_X, means_array))
    ## Dropping the first column
    formatting_matrix_X = np.delete(word_embedding_X, 0, 0)
    return formatting_matrix_X


def word_embedding(df):
    
    articles = df['Article Text']
    general_vectors_distr = []
    mem_l = []
    nb_words = 0
    iterat1 = 0
    for art in articles:
        tokens = nltk.word_tokenize(art, language='italian')
        for t in tokens:
            try:
                word_vector = glove_model.wv[t]
                general_vectors_distr.append(word_vector)
                if t in mem_l:
                    pass
                else:
                    mem_l.append(t)
                    nb_words += 1
            except:
                pass
        # print(iterat1)
        iterat1 += 1
      
    all_embs = np.stack(general_vectors_distr)  
    emb_mean, emb_std = all_embs.mean(), all_embs.std()    
    embed_size = all_embs.shape[1]
    
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    ## filling the embedding matrix
    iterat2 = 0
    for token, i in zip(mem_l, range(nb_words)):
        embedding_matrix[i] = glove_model[token]
        # print(iterat2)
        iterat2 += 1
    
    return embedding_matrix

word_embedding_starting_df = word_embedding(pd.concat([starting_train, starting_test])) ## Loading Word Embeddings
# pd.DataFrame(word_embedding_starting_df).to_excel('Word Embedding Starting Dataset.xslx')
starting_train_X = starting_train['Article Text']
starting_test_X = starting_test['Article Text']
starting_train_Y = starting_train['Sentiment']
starting_test_Y = starting_test['Sentiment']


embed_size = 300 # how big is each word vector
max_features = 12022 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in an article to use

## To input into the neural network, we must first tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(starting_train_X))
train_X = tokenizer.texts_to_sequences(starting_train_X)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(starting_test_X))
test_X = tokenizer.texts_to_sequences(starting_test_X)

## Pad the sequences
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Converting the output shape to categoricals
from keras.utils import to_categorical

## converting the bad sentiment (i.e. -1, to 2)

starting_train_Y[starting_train_Y == -1] = 2

train_Y = to_categorical(starting_train_Y)

embedding_matrix = np.mean([word_embedding_starting_df], axis = 0)
np.shape(embedding_matrix)
model = model_bilstm_attention(embedding_matrix)
model.summary()
## Fitting the model with 1 epoch and with dummy for 0 sentiment
## Given the initial distribution of sentiments : <200 for -1 and approx. 200 for class 1
## I will use class weight function
class_weight = {0 : 1.15,
                1 : 0.85}
class_weight_negatives = {0 : 1.,
                          1 : 2.4}
class_weight_positives = {0 : 1.,
                          1 : 1.8}
model.fit(train_X, train_Y[:, 2], validation_split=0.2, epochs=20, class_weight = class_weight_negatives)
pred_train_x = model.predict(train_X)
pred_train_x_ones = model.predict(train_X)
pred_train_x_negatives = model.predict(train_X)
pred_test_x = model.predict(test_X)
pred_test_x_ones = model.predict(test_X)
pred_test_x_negatives = model.predict(test_X)

## compute the first difference from predicted values and actual values in train
pred_train_x_round = np.round(pred_train_x, decimals=0)

ones = np.ones(len(pred_train_x))
indexes = pred_train_x_round.flatten() == train_Y[:, 0]
count_predicted = np.sum(ones[indexes])

perc_predicted = count_predicted / train_Y[:,0].shape[0]

print(f'% of predicted articles Train : {perc_predicted*100}%')

pred_train_x_ones_round = np.round(pred_train_x_ones, decimals=0)
indexes_ones = pred_train_x_ones_round.flatten() == train_Y[:, 1]
count_predicted_ones = np.sum(ones[indexes_ones])

perc_predicted_ones = count_predicted_ones / train_Y[:,1].shape[0]
print(f'% of positive predicted articles Train : {perc_predicted_ones*100}%')

pred_train_x_negatives_round = np.round(pred_train_x_negatives, decimals=0)
indexes_negatives = pred_train_x_negatives_round.flatten() == train_Y[:, 1]
count_predicted_negatives = np.sum(ones[indexes_negatives])

perc_predicted_negatives = count_predicted_negatives / train_Y[:,1].shape[0]
print(f'% of positive predicted articles Train : {perc_predicted_negatives*100}%')

pred_test_X_round = np.round(pred_test_x, decimals=0)
pred_test_X_ones_round = np.round(pred_test_x_ones, decimals=0)
pred_test_X_negatives_round = np.round(pred_test_x_negatives, decimals=0)

## Starting the iteration for at least 40000 articles.
## From there on the remaining ones will be predicted normally

train_pred_Y = np.hstack((pred_train_x_round, pred_train_x_ones_round, pred_train_x_negatives_round))
test_pred_Y = np.hstack((pred_test_X_round, pred_test_X_ones_round, pred_test_X_negatives_round))

starting_df_Y = np.vstack((train_Y, test_pred_Y))


## getting last_index

last_ind = starting_test.index[-1]
iterat = 0
df_Y_i = starting_df_Y
starting_df_iterat = final_df.iloc[:last_ind+1, :]
accuracy_l = []
accuracy_ones_l = []
accuracy_negatives_l = []
validation_l = []
validation_ones_l = []
validation_negatives_l = []
n = 0

class_weight = {0 : 4.0,
                1 : 1.0}
class_weight_negatives = {0 : 1.,
                          1 : 5.0}
class_weight_positives = {0 : 1.,
                          1 : 4.5}

while n < 40000:
    
    starting_train_iterat, starting_test_iterat = train_test_division(starting_df_iterat, final_df, (0.7, 0.3))
    
    df_iterat = pd.concat([starting_train_iterat, starting_test_iterat])
    n = df_iterat.shape[0]
    
    if n >= 10000:
        print(f'Do you want to continue? ({n} articles classified)')
        inp = input()
        if inp == 'Y':
            break
        else:
            pass
    
    try:
        word_embedding_df_i = word_embedding(pd.concat([starting_train_iterat, starting_test_iterat]))
    except:
        pass
    max_features = word_embedding_df_i.shape[0]
    
    starting_train_X_i = starting_train_iterat['Article Text']
    starting_test_X_i = starting_test_iterat['Article Text']
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(starting_train_X_i))
    train_X_i = tokenizer.texts_to_sequences(starting_train_X_i)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(starting_test_X_i))
    test_X_i = tokenizer.texts_to_sequences(starting_test_X_i)
    
    print(f'df iterat initial shape : {df_iterat.shape}')
    ## Pad the sequences
    train_X_i = pad_sequences(train_X_i, maxlen=maxlen)
    print(f'Shape Train X : {train_X_i.shape}')
    test_X_i = pad_sequences(test_X_i, maxlen=maxlen)
    print(f'Shape Test X : {test_X_i.shape}')
    print(f'Shape Y : {df_Y_i.shape}')
    
    embedding_matrix = np.mean([word_embedding_df_i], axis = 0)
    
    model = model_bilstm_attention(embedding_matrix)
    
    for i in range(3):
        if i == 0:
            res_i_ones = model.fit(train_X_i, df_Y_i[:, i], validation_split=0.2, epochs=10, class_weight = class_weight)
            
            acc_i = res_i_ones.history['accuracy'][-1]
            val_i = res_i_ones.history['val_accuracy'][-1]
            accuracy_l.append(acc_i)
            validation_l.append(val_i)
            pred_test_x_i = model.predict(test_X_i)
            pred_test_X_round_i = np.round(pred_test_x_i, decimals=0)

            
            
        elif i == 1:
            res_i_pos = model.fit(train_X_i, df_Y_i[:, i], validation_split=0.2, epochs=10, class_weight = class_weight_positives)
            
            acc_i = res_i_pos.history['accuracy'][-1]
            val_i = res_i_pos.history['val_accuracy'][-1]
            accuracy_ones_l.append(acc_i)
            validation_ones_l.append(val_i)
            pred_test_x_ones_i = model.predict(test_X_i)
            pred_test_X_ones_round_i = np.round(pred_test_x_ones_i, decimals=0)

            
        elif i == 2:
            res_i_neg = model.fit(train_X_i, df_Y_i[:, i], validation_split=0.2, epochs=10, class_weight = class_weight_negatives)
            
            acc_i = res_i_neg.history['accuracy'][-1]
            val_i = res_i_neg.history['val_accuracy'][-1]
            accuracy_negatives_l.append(acc_i)
            validation_negatives_l.append(val_i)
            pred_test_x_negatives_i = model.predict(test_X_i)
            pred_test_X_negatives_round_i = np.round(pred_test_x_negatives_i, decimals=0)
    
    print(f'prevision shape : {pred_test_X_round_i.shape}')
    print(f'prevision shape ones : {pred_test_X_ones_round_i.shape}')
    print(f'prevision shape negat : {pred_test_X_negatives_round_i.shape}')
    
    test_pred_Y_i = np.hstack((pred_test_X_round_i, pred_test_X_ones_round_i, pred_test_X_negatives_round_i))
    df_Y_i = np.vstack((starting_df_Y, test_pred_Y_i))
    
    starting_df_Y = df_Y_i
    starting_df_iterat = df_iterat
    print(f'df iterat final shape : {starting_df_iterat.shape}')
    print(f'final df Y shape : {df_Y_i.shape}')
    
    iterat += 1
    
## 5 + 1 iterations reached before an error on the input sizes. Let us plot the first 3 iterations with accuracy test, accuracy train

import matplotlib.pyplot as plt
  
iterat = list(range(1, 7))    

data = ((iterat, accuracy_l), (iterat, validation_l), (iterat, accuracy_ones_l), 
        (iterat, validation_ones_l), (iterat, accuracy_negatives_l), (iterat, validation_negatives_l))
colors = ('blue', 'red',
          'green', 'black',
          'yellow', 'brown')
groups = ('Accuracy Train 0', 'Accuracy Test 0', 'Accuracy Train 1', 'Accuracy Test 1', 'Accuracy Train -1', 'Accuracy Test -1')

fig= plt.figure(figsize=(30,15))
ax = fig.add_subplot(1, 1, 1)

for dat, color, group in zip(data, colors, groups):
    x, y = dat
    ax.plot(x, y, c=color, label=group)

plt.legend(loc=2)
plt.xlabel('Iteration')
plt.ylabel('Accuracy Train and Test')
plt.title('Result First Dynamic Test RNN')
plt.show()