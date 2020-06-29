#%%
import pandas as pd
import numpy as np
import os
print(os.listdir())
from __future__ import absolute_import, division, print_function,unicode_literals

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
#import skimage not running in py 3.8
import keras
from sklearn.preprocessing import LabelEncoder

# %%
data = pd.read_csv('data/IMDB Dataset.csv')
data.info()
data.sentiment.value_counts()
# %%
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])


# %%
#avg number of words per sample
plt.figure(figsize=(10,6))
plt.hist([len(i) for i in list(data['review'])],50)
plt.xlabel('Len of samples')
plt.ylabel('num of samples')
plt.title('sample len distribution')
plt.show()

# %%
