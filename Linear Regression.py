#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


df = pd.read_csv('Linear-Regression-Data.csv')
df.head()


# In[15]:


import numpy as np
import pandas as pd
import keras
import keras.backend as kb
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])


# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
x = df.x
y = df.y
x = x.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=42)


# In[5]:


optimizer = tf.keras.optimizers.RMSprop(0.0099)
model.compile(loss='mean_squared_error',optimizer=optimizer)
model.fit(x,y,epochs=500)


# In[24]:


model.predict([200])


# In[25]:


sns.scatterplot(x='x',y='y', data=df)


# In[ ]:




