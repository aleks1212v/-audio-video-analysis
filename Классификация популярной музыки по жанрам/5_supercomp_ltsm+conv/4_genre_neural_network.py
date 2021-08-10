# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:55:57 2020

@author: aleks1212
"""


import os
import pandas as pd
import numpy as np
from  pathlib import Path
from  sklearn.preprocessing  import StandardScaler 
from  sklearn.model_selection import train_test_split

from  tqdm  import tqdm 
from  keras.models  import Sequential 
from  keras.layers  import LSTM, Dense, Dropout, Activation 
from  keras.optimizers  import Adam
from  keras.callbacks import ModelCheckpoint
from  keras import backend as K

labels = ['allrock', 'blues', 'country', 'electronic', 'hip-hop', \
          'jazz', 'latin-american', 'pop', 'r-n-b', 'reggae', 'shanson', 'world']

labels_small = ['allrock_small', 'blues_small', 'country_small', 'electronic_small', 'hip-hop_small', \
          'jazz_small', 'latin-american_small', 'pop_small', 'r-n-b_small', 'reggae_small', 'shanson_small', 'world_small']

x_lim = 431  #(22050/512) * 10 = 431 отсчет на 10 секунд звучания
y_lim = 15   #n_mfcc

n_songs = 30 #песен у каждого исполнителя
n_artists = 300 #исполнителей в каждом жанре. Но может быть чуть меньше.


#записать все данные в один файл

#filepath_out = '431na15_full.csv'
song = pd.DataFrame()
print('Загрузка:')
for label in labels:
   print(label)
   file = label + '.csv'
   sound = pd.read_csv(file) #читаем файл с mfcc конкретного жанра
   sound['class'] = Path(file).stem
   song = pd.concat([song, sound], ignore_index = True)
   #song.to_csv(filepath_out, index=False, mode = 'a')
            
print(song.shape)

song = song.sample(frac=1).reset_index(drop=True) 
X = song[[str(i) for i in range(y_lim)]]
y = song['class']

#преобразование one-hot и нормализация
tran = StandardScaler()
X = tran.fit_transform(X)
y = pd.get_dummies(y, columns = labels)
print(y.shape)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.20, random_state=42) #тестовая
X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, random_state=42) #валидационная

n_row = (X.shape[0] // x_lim) * x_lim
new_row = n_row // x_lim
X = X[:n_row, :]
X = np.array(X).reshape(new_row, x_lim, y_lim) #для подачи на входд LTSM
y = y.iloc[:n_row, :]
y = np.array(y).reshape(new_row, x_lim, 12)[:, 0, :]
print(X.shape)
print(y.shape)

n_row = (X_val.shape[0] // x_lim) * x_lim
new_row = n_row // x_lim
X_val = X_val[:(X_val.shape[0] // x_lim) * x_lim, :]
X_val = np.array(X_val).reshape(-1, x_lim, y_lim) #для подачи на входд LTSM
y_val = y_val.iloc[:n_row, :]
y_val = np.array(y_val).reshape(new_row, x_lim, 12)[:, 0, :]
print(X_val.shape)
print(y_val.shape)

n_row = (X_test.shape[0] // x_lim) * x_lim
new_row = n_row // x_lim
X_test = X_test[:(X_test.shape[0] // x_lim) * x_lim, :]
X_test = np.array(X_test).reshape(-1, x_lim, y_lim) #для подачи на входд LTSM
y_test = y_test.iloc[:n_row, :]
y_test = np.array(y_test).reshape(new_row, x_lim, 12)[:, 0, :]
print(X_test.shape)
print(y_test.shape)

# Метрики 
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
           

model = Sequential()

model.add(LSTM(256, return_sequences=False, input_shape=(x_lim, y_lim,), activation = 'tanh'))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(12, activation = 'softmax'))


model.compile(loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1], optimizer='adam')


checkpoint = ModelCheckpoint("lstm-genres-full-{epoch:02d}-{val_f1:.2f}.hdf5", monitor='val_f1', save_best_only=True, mode='max', period=1)
history = model.fit(X, y, batch_size=128, epochs=1000, validation_data=(X_val, y_val), callbacks = [checkpoint])


y_pred = model.predict_classes(X_test) #To predict labels
precision_test = precision(y_test, y_pred)
recall_test = recall(y_test, y_pred)
f1_test = f1(y_test, y_pred)
print(f"Precision test: {precision_test}")
print(f"Recall test: {recall_test}")
print(f"f1 test: {f1_test}")




