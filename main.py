#%%
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds

#%%
df = pd.read_csv('data\IMDB Dataset.csv')
sns.countplot(df['sentiment'])
#%%
le = LabelEncoder()
x_rev_train,x_rev_test,y_label_train,y_label_test = train_test_split(df['review'].values, df['sentiment'].values,test_size=0.2)
y_label_train = le.fit_transform(y_label_train)
y_label_test = le.fit_transform(y_label_test)

#%%
#preprocessing the text
tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(x_rev_train)
word_index = tokenizer.word_index

#%%
training_sequence = tokenizer.texts_to_sequences(x_rev_train)
testing_sequence = tokenizer.texts_to_sequences(x_rev_test)
train_pad_sequence = pad_sequences(training_sequence,maxlen=200,
                    truncating='post',padding='pre')
#%%
#glove vectors for embedding

