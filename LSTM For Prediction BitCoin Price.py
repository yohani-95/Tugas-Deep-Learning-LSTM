#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[2]:


data = pd.read_csv("bitcoin_ticker.csv") #import data


# Data yang digunakan berasar dari kaggel, url: https://github.com/sudharsan13296/Bitcoin-price-Prediction-using-LSTM .
# Data tersebut tentang Prediksi harga bitcoin secara time series dari berbagai makret penyedia layanan bitcoin. 
# variabel yang digunakan pada data ini yaitu 'last' (harga jual bitcoin sesuai harga mata uang tiap masing-masing negara. Harga tersebut berbeda-beda tergantung market penyedianya).

# In[3]:


data.head()


# In[4]:


data['rpt_key'].value_counts() 


# In[5]:


df = data.loc[(data['rpt_key'] == 'btc_usd')]


# In[6]:


df.head()


# In[30]:


df = data.loc[(data['rpt_key'] == 'btc_krw')]


# In[31]:


df.head()


# In[7]:


df = df.reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime_id'])
df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]


# In[8]:


df = df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]


# In[9]:


df.head()


# In[10]:


df = df[['last']]


# In[11]:


dataset = df.values
dataset = dataset.astype('float32')


# In[12]:


dataset


# Selanjutnya, dilakukan normalisasi menggunakan fungsi sigmoid atau tanh activation. 
# Dilakukan normalisasi agar nilai yang digunakan pada kesalahan rata-rata tidak terlalu besar, nilai harus berkisar antara rang 0, 1

# In[13]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[14]:


dataset


# In[15]:


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# In[16]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)


# In[17]:


look_back = 10
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)


# In[18]:


trainX


# In[19]:


trainY


# In[20]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Selanjutnya, kami menggunakan metode LSTM untuk melakukan analisa dengan menggunakan data train, units = 4, epoch=100, batch_size=256, optimizer menggunakan adam.

# In[21]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=2)


# 

# In[22]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[23]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[24]:


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


# Dengan melakukan epoch=100 dan validation_size=1500, hasil prediksi yang didapatkan dengan menggunakan model LSTM mampu memprediksikan akurasi pada data train sebesar 5.04 (0.504) sedangkan data test sebesar 5.92 (0.592) dengan lost_error=8.4152e-05

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




