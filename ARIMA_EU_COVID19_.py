#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.arima_model import ARMA


# In[21]:


# Import modules
# Load in the time series
df = pd.read_excel('eu_data_fix2020.xlsx')
df.head(10)


# In[22]:


df.period


# In[23]:


#if u have another data with a lot colum and want to clean
data = df.dropna()
data.index = pd.to_datetime(data.period)
data = data["EU"]['2018-01-01':'2020-10-01']


# In[26]:


data.describe()


# In[27]:


# Data Exploration

plt.figure(figsize=(16,7))
fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.set_xlabel('Period')
ax1.set_ylabel('Number of people confirmed COVID-19')
ax1.plot(data)


# # Checking Stationarity
# * Method 1 - Rolling Statistics
# * Method 2 - Dickey Fuller Test

# In[28]:


# METHOD 1
#Determining rolling statistics
# WINDOW = 12 -- number of month, if days use 365
rolmean = data.rolling(window=12).mean()
rolstd = data.rolling(window=12).std()

plt.figure(figsize=(16,7))
fig = plt.figure(1)


# In[29]:


#plot rolling statistics :

orig = plt.plot(data,color='blue',label='Original') #originaldata
mean = plt.plot(rolmean,color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='green',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standar Deviation')
plt.show(block=False)


# We can see from data above
# * Rolling standar deviation (green) more or less constant over the time 
# but rolling mean (red) is not constant
# so we can say this data series in <b>not stationary</b>

# In[34]:


# METHOD 2
# Perform Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey-Fuller Test :')
dftest = adfuller(df["EU"], autolag='AIC') #AIC Akaike Info Criteria

dfoutput = pd.Series(dftest[0:4], index=['Test statistics','p-value','#Lags Used','Number of Observation Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value(%s)'%key] = value
    
print(dfoutput)


# ### 0 hipotesis = nonstasionary,
# to make it stasionary it should pass these criterias 
# * p value < 0,05 
# * critical value should > Test statistic
# 
# We can see from data above, we have hight p-value 0.126642 , it's higer than 0.05
# Critical value < Test statistic
#    ###### so we can not reject null hypothesis - the data is non statsionary

# # MAKING Series Stationary

# In[36]:


#Let's transform this LOG
plt.figure(figsize=(16,7))
fig = plt.figure(1)

import numpy as np
ts_log = np.log(data)
plt.plot(ts_log)


# In[37]:


#DECOMPOSITION -- this decomposition will give us difference component of our time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=1,model='multiplicative')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,8))
fig = plt.figure(1)

plt.subplot(411)
plt.plot(ts_log,label ='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label ='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label ='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# #### From the data above we can see even after log transformation it's still non-stationary

# In[38]:


#Let's Try differencing --- we will SHIFT timeseries by 1 and substact timeseries by original series

plt.figure(figsize=(16,8))
fig = plt.figure(1)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

# METHOD 1
#Determining rolling statistics using DIFFERENCE
# WINDOW = 12 -- number of month, if days use 365
rolmean = ts_log_diff.rolling(window=12).mean()
rolstd = ts_log_diff.rolling(window=12).std()


#plot rolling statistics DIFFERENCE :
orig = plt.plot(ts_log_diff,color='blue',label='Original') #originaldata
mean = plt.plot(rolmean,color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='green',label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standar Deviation')
plt.show(block=False)


# #### We can see from data above
# * Blue - our difference to timeseries
# * Rolling standar deviation (green) & rolling mean (red) are now no upward pattern in the mean also in standar deviation.
# so by the definition of stationarity , if i take mean by these 2 point , there will not be too much difference between these . So I can say/assume now <b>this time series will be stationary time series</b>

# # ACF AND PACF PLOT

# In[39]:


data.sort_index(inplace=True)


# In[40]:


from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=10)  
lag_pacf = pacf(ts_log_diff, nlags=10)


# In[41]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_log_diff.dropna(),lags=15,ax=ax1) #nlags should be < 50% of sample size our observarion 34 so should be < 16

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_log_diff.dropna(),lags=15,ax=ax2)


# * the thumb rule is area with highligthed called confidence intervals
# * The first line crosses this chart is the line of our order
# * AR-auto regression ( Auto correlation ) , p = 1 #Inside highlegthed conv interval
# * MA-moving average (Parcial Autocorellation ), q = 1
# * d = 1 
# 
# because we don'nt know the combination we just try

# # MAKING ARIMA MODEL

# In[42]:


# Import the ARMA model
from statsmodels.tsa.arima_model import ARIMA

plt.figure(figsize=(16,8))
# Instantiate the model ARMA(p=1,d=0,q=1)
model = ARIMA(ts_log, order=(1,1,0))
results_ARIMA = model.fit(disp=-1)


plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
print('Plotting ARIMA model')


# # Taking result to original Scale

# In[43]:


ARIMA_diff_predictions = pd.Series(results_ARIMA.fittedvalues,copy=True)
print(ARIMA_diff_predictions.head())


# In[44]:


ARIMA_diff_predictions_cumsum = ARIMA_diff_predictions.cumsum()
print(ARIMA_diff_predictions_cumsum.head())


# In[45]:


ARIMA_log_prediction = pd.Series(ts_log.iloc[0],index=ts_log.index)
ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions_cumsum,fill_value=0)
ARIMA_log_prediction.head()


# In[46]:


plt.figure(figsize=(12,8))
predictions_ARIMA = np.exp(ARIMA_log_prediction)
plt.plot(data)
plt.plot(predictions_ARIMA)
plt.title('RMSE : %.4f'% np.sqrt(sum((predictions_ARIMA-data)**2)/len(data)))


# * Blue is original
# * Orange is prediction 

# In[47]:


results_ARIMA.predict(10,20)


# In[49]:


get_ipython().system('pip install pmdarima')


# In[50]:


import pmdarima as pm
def arimamodel(timeseries):
    automodel = pm.auto_arima(timeseries,
                             start_p=3,
                             start_q=3,
                             max_p=5,
                             max_q=5,
                             test="adf",
                             seasonal=True,
                             trace=True)
    return automodel


# In[51]:


arimamodel(ts_log)


# ### So the best combination showing from arima is 2,2,2

# In[66]:


# Import the ARMA model
from statsmodels.tsa.arima_model import ARIMA

plt.figure(figsize=(16,8))
# Instantiate the model ARMA(p=2,d=2,q=0)
model = ARIMA(ts_log, order=(2,2,0))
results_ARIMA = model.fit(disp=-1)


plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - ts_log_diff["EU"])**2))
print('Plotting ARIMA model')


# After trying combination we get the combination 2,2,1 is best result

# In[67]:


print(results_ARIMA.summary())


# In[68]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head())


# In[72]:


#CONVERT TO CUMULATIVE SUM
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head)


# In[79]:


predictions_ARIMA_log = pd.Series(ts_log[1],index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[81]:


plt.figure(figsize=(12,8))
predictions_ARIMA= np.exp(predictions_ARIMA_log)
plt.plot(data)
plt.plot(predictions_ARIMA)


# In[82]:


ts_log


# In[88]:


# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.arima_model import ARMA


# In[104]:


plt.figure(figsize=(25,20))
fig = plt.figure(1)
#result ARIMA ( I want to predict 3 year , 34 + 36 = 70 )
results_ARIMA.plot_predict(10,70)
x=results_ARIMA.forecast(steps=36)


# In[ ]:





# In[ ]:




