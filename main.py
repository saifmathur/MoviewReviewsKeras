#%%
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds


df = pd.read_csv('data\IMDB Dataset.csv')
sns.countplot(df['sentiment'])

le = LabelEncoder()
x_rev_train,x_rev_test,y_label_train,y_label_test = train_test_split(df['review'].values, df['sentiment'].values,test_size=0.2)
y_label_train = le.fit_transform(y_label_train)
y_label_test = le.fit_transform(y_label_test)


#preprocessing the text
tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(x_rev_train)
word_index = tokenizer.word_index


training_sequence = tokenizer.texts_to_sequences(x_rev_train)
testing_sequence = tokenizer.texts_to_sequences(x_rev_test)
train_pad_sequence = pad_sequences(training_sequence,maxlen=200,
                    truncating='post',padding='pre')
test_pad_sequence = pad_sequences(training_sequence,maxlen=200,truncating='post',padding='pre')

#glove vectors for embedding
embedded_words = {}
with open('data/glove.6B.200d.txt',encoding='utf8') as file:
    for line in file:
        words,coeff = line.split(maxsplit=1)
        coeff = np.array(coeff.split(),dtype=float)
        embedded_words[words] = coeff


embedding_matrix = np.zeros((len(word_index) + 1,200))
for word, i in word_index.items():
    embedding_vector = embedded_words.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#%%
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Embedding

model = Sequential()
model.add(Embedding(len(word_index)+1,200,weights =[embedding_matrix],input_length=200))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy',metrics='accuracy')
history = model.fit(train_pad_sequence,y_label_train,epochs=35)



