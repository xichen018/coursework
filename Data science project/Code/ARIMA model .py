#!/usr/bin/env python
# coding: utf-8

# Using ARIMA model to predict the stock index price

# In[1]:


from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt         #for visualization
import seaborn as sns    #for visualization


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.arima_model import ARIMA
from pandas_datareader.data import DataReader


# In[2]:



#Dow Jones U.S. Financial Services Index (USD)
#Dow Jones U.S. Health Care Index      

dataset = pd.read_excel("D:/3_index.xlsx")

    


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


# drop the unnecessary columns(S&P500 index data drop)
dataset.drop(['PX_OPEN','LAST_PRICE','PX_OPEN.1','PX_OPEN.2'],inplace=True, axis=1)


# In[6]:


dataset.head()


# In[7]:


#Dow Jones U.S. Financial Services Sector Index(DJUSGF)
#Dow Jones U.S. Health Care Index(DJUSHC)

dataset.rename(columns={"LAST_PRICE.1":'DJUSGF',
               'LAST_PRICE.2':'DJUSHC'},inplace=True)


# In[8]:


dataset.drop(index=[0])


# In[9]:


dataset.describe()


# In[10]:


dataset.info()


# In[11]:


dataset.set_index("Dates",inplace=True)


# In[12]:


FinancialService_data_train=dataset.loc["2002-08-01":"2007-01-30","DJUSGF"]
HealthCare_data_train=dataset.loc["2002-08-01":"2007-01-30",'DJUSHC']


# In[13]:


FinancialService_data_train.head()


# In[14]:


FinancialService_data_train.tail()


# In[15]:


HealthCare_data_train.head()


# In[16]:


HealthCare_data_train.tail()


# In[17]:


sectors=[FinancialService_data_train,HealthCare_data_train]
sectors_name=['FinancialService_data_train','HealthCare_data_train']
plt.figure(figsize=(10,200))
plt.subplots_adjust(top=1.25,bottom=1.2)

for i, sector in enumerate(sectors,1):
    plt.subplot(2,1,i)
    plt.plot(sector)
    plt.ylabel('Price')
    plt.title(f"{sectors_name[i - 1]}")
plt.show()    


# In[18]:


# Daily return for two indices

plt.figure(figsize=(20, 5))


for i, sector in enumerate(sectors, 1):
    plt.subplot(1, 2, i)
    sns.distplot((sector.pct_change()).dropna(), bins=100, color='purple')
    plt.ylabel('Daily Return')
    plt.title(f'{sectors_name[i - 1]}')



# In[19]:


#Index price corrlation 
plt.figure(figsize=(6,6))
sns.heatmap(dataset.corr(), annot=True, cmap='summer')


# Before applying any statistical model on a time series, we want to ensure it’s stationary
# There are two primary way to determine whether a given time series is stationary:
# 
# 1) Rolling Statistics: Plot the rolling mean and rolling standard deviation. 
#    The time series is stationary if they remain constant with time (with the naked eye look to see if the lines are     straight and parallel to the x-axis).
#    
# 2) ADCF Test - Augmented Dickey–Fuller test: it is used to gives us various values that can help in identifying stationarity. The Null hypothesis says that a TS is non-stationary. It comprises of a Test Statistics & some critical values for some confidence levels. If the Test statistics is less than the critical values, we can reject the null hypothesis & say that the series is stationary. THE ADCF test also gives us a p-value. Acc to the null hypothesis, lower values of p is better.
# 
# 
# 
# 
# 

# In[20]:


# Rolling Statistics

FS_rolling_mean=FinancialService_data_train.rolling(window = 50).mean()
FS_rolling_std=FinancialService_data_train.rolling(window = 50).std()

HC_rolling_mean=HealthCare_data_train.rolling(window = 50).mean()
HC_rolling_std=HealthCare_data_train.rolling(window = 50).std()
   

plt.figure(figsize=(10,400))
plt.subplots_adjust(top=1.25,bottom=1.2)


plt.subplot(2,1,1)
plt.plot(FS_rolling_mean, color='red', label='Rolling Mean') 
plt.plot(FS_rolling_std, color='blue',label='Rolling Std')
plt.plot(FinancialService_data_train, color='black', label='Original') 
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation for FinancialService train Index') 

plt.subplot(2,1,2)
plt.plot(HC_rolling_mean, color='red', label='Rolling Mean') 
plt.plot(HC_rolling_std, color='blue',label='Rolling Std')
plt.plot(HealthCare_data_train, color='black', label='Original') 
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation for HealthCare train Index') 



# The rolling mean increase with time rolling but the standard deviation is fairly constant with time. Therefore, we can conclude that the time series is not stationary. Further test:
# 

# In[21]:


#Perform Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
FS_result = adfuller(FinancialService_data_train)
HC_result = adfuller(HealthCare_data_train)

print("For Financial service sector:")
print('ADF Statistic: {}'.format(FS_result[0])) 
print('p-value: {}'.format(FS_result[1])) 

print("For Health Care sector:")
print('ADF Statistic: {}'.format(HC_result[0])) 
print('p-value: {}'.format(HC_result[1])) 



# the p-value is greater than the threshold (0.05). Thus, we can conclude that the time series is not stationary.

# In[22]:


#Create a function to test stationary:
def test_stationarity(data,name):
  # rolling statistics
    rolling_mean = data.rolling(window=50).mean() 
    rolling_std = data.rolling(window=50).std()
  # rolling statistics plot
    plt.plot(rolling_mean, color='red', label='Rolling Mean') 
    plt.plot(rolling_std, color='blue',label='Rolling Std')
    plt.plot(data, color='black', label='Original') 
    plt.legend(loc='best')
    plt.title(f'Rolling Mean & Standard Deviation for {name}')
  # Dickey–Fuller test:
    result = adfuller(data)
    print(f"For {name} : ")
    print('ADF Statistic: {}'.format(result[0])) 
    print('p-value: {}'.format(result[1])) 


# #Data Transformation to achieve Stationarity (Two ways)

# In[23]:


#1 Log Scale Transformation

def log_scale_trans (data,i,name):   
    data_log = np.log(data) 
    plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    plt.plot(data_log)
    plt.title( f"{name} log ")
    rolling_mean = data_log.rolling(window=50).mean() 
    data_log_minus_mean = data_log - rolling_mean 
    data_log_minus_mean.dropna(inplace=True)
    plt.subplot(1,3,2)
    plt.plot(data_log_minus_mean)
    plt.title(f"{name}log minus mean")
    plt.subplot(1,3,3)
    test_stationarity(data_log_minus_mean,name)


# In[24]:


for i,sector in enumerate(sectors,1):
   
    log_scale_trans(sector,i,sectors_name[i-1])
    plt.tight_layout()


# From the charts above, p-values are small than 0.05,Therefore, we have suffcient evidence to say the time series is stationary.

# In[25]:


#2 Time Shift Transformation


FS_shift = FinancialService_data_train - FinancialService_data_train.shift() 
FS_shift.dropna(inplace=True)
plt.figure(figsize=(20,10))
test_stationarity(FS_shift,sectors_name[0])




# In[26]:



HC_shift = HealthCare_data_train - HealthCare_data_train.shift() 
HC_shift.dropna(inplace=True)
plt.figure(figsize=(20,10))
test_stationarity(HC_shift,sectors_name[1])


# From the charts and p-value obtained above, we have suffcient evidence to say the time series is stationary.

# AutoRegressive Integrated Moving Average Model (ARIMA)
# 
# Three integers (p, d, q) are typically used to parametrize ARIMA models.
# p: number of autoregressive terms (AR order)
# d: number of nonseasonal differences (differencing order) 
# q: number of moving-average terms (MA order)

# In[27]:


#Determine P
#Auto Correlation Function (ACF)
#The correlation between the observations at the current point in time and the observations at all previous points in time. 
#We can use ACF to determine the optimal number of MA terms. 


# In[28]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#Plot ACF:


plt.figure(figsize=(10,10))
plt.subplot(211)
#Financial Service data
plot_acf(FS_shift, ax=plt.gca())
plt.subplot(212)
#Health Care data
plot_acf(HC_shift, ax=plt.gca())
plt.show()


# From the charts above,
# we choose p=8 for financial Service data
# choose p=8 for Health Care data
# 

# In[29]:


#Determie q
#Partial Auto Correlation Function (PACF)
#As the name implies, PACF is a subset of ACF. 
#PACF expresses the correlation between observations made at two points in time while accounting for any influence from other data points. 
#We can use PACF to determine the optimal number of terms to use in the AR model.

#Plot PACF:


plt.figure(figsize=(10,10))
plt.subplot(211)
#Financial Service data
plot_pacf(FS_shift, ax=plt.gca())
plt.subplot(212)
#Health Care data
plot_pacf(HC_shift, ax=plt.gca())
plt.show()







# From the charts above,
# we choose q=8 for financial Service data
# choose q=3 for Health Care data
# 

# In[30]:


#For Financial Service index data



# Build an ARIMA model with AR of order 8, differencing of order 1 and MA of order 8.
FS_model = ARIMA(FinancialService_data_train, order=(8,1,8))
fitted_FS = FS_model.fit(disp=-1)


# In[31]:


print(fitted_FS.summary().tables[1])


# In[32]:


#For Health Care index data



# Build an ARIMA model with AR of order 8, differencing of order 1 and MA of order 3.
HC_model = ARIMA(HealthCare_data_train, order=(8,1,3))
fitted_HC = HC_model.fit(disp=-1)


# In[33]:


print(fitted_HC.summary().tables[1])


# Use period 2007-01-02 to 2007-08-01 testing the accuracy 

# In[34]:


#Obtain test set (actual stock price)
FS_test=dataset.loc["2007-01-31":"2007-09-04","DJUSGF"]
HC_test=dataset.loc["2007-01-31":"2007-09-04",'DJUSHC']


# In[35]:


#Obtain crisis
FS_crisis=dataset.loc['2007-09-05':'2008-03-04','DJUSGF']
HC_crisis=dataset.loc['2007-09-05':'2008-03-04','DJUSHC']


# In[ ]:





# In[36]:


FS_test.describe()


# In[37]:


HC_test.describe()


# In[46]:


# FOR Financial Serivices Sector:


# Forecast 
FS_fc, FS_se, FS_conf = fitted_FS.forecast(155, alpha=0.05)  # 95% confidence interval
FS_fc2, FS_se2, FS_conf2 = fitted_FS.forecast(130, alpha=0.05)



# Make as pandas series
FS_fc_series = pd.Series(FS_fc, index=FS_test.index)
FS_lower_series = pd.Series(FS_conf[:, 0], index=FS_test.index)
FS_upper_series = pd.Series(FS_conf[:, 1], index=FS_test.index)



FS_fc_series2 = pd.Series(FS_fc2, index=FS_crisis.index)
FS_lower_series2 = pd.Series(FS_conf2[:, 0], index=FS_crisis.index)
FS_upper_series2 = pd.Series(FS_conf2[:, 1], index=FS_crisis.index)




# Plot
plt.figure(figsize=(12,5), dpi=100)
#plt.plot(FinancialService_data_train, label='training')
#plt.plot(FS_test, label='actual')
plt.plot(FS_crisis, label='crisis')
#plt.plot(FS_fc_series, label='forecast')
plt.plot(FS_fc_series2, label='forecast in crisis')
#plt.fill_between(FS_lower_series.index, FS_lower_series, FS_upper_series, 
              #   color='k', alpha=.15)
plt.fill_between(FS_lower_series2.index, FS_lower_series2, FS_upper_series2, 
                 color='k', alpha=.15)
plt.title('Forecast VS Actuals For Financial Service')
plt.legend(loc='upper left', fontsize=8)
plt.show()
            


# In[39]:


pip install RegscorePy


# In[40]:


# Calculate AIC for the predicted data and actual data
from RegscorePy import *
aic.aic(FS_fc_series, FS_test, 17)


# In[47]:


# Forecast
HC_fc, HC_se, HC_conf = fitted_HC.forecast(155, alpha=0.05)  # 95% confidence interval

HC_fc2, HC_se2, HC_conf2 = fitted_HC.forecast(130, alpha=0.05) 




# Make as pandas series
HC_fc_series = pd.Series(HC_fc, index=HC_test.index)
HC_lower_series = pd.Series(HC_conf[:, 0], index=HC_test.index)
HC_upper_series = pd.Series(HC_conf[:, 1], index=HC_test.index)


HC_fc_series2 = pd.Series(HC_fc2, index=HC_crisis.index)
HC_lower_series2 = pd.Series(HC_conf2[:, 0], index=HC_crisis.index)
HC_upper_series2 = pd.Series(HC_conf2[:, 1], index=HC_crisis.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
#plt.plot(HealthCare_data_train, label='training')
#plt.plot(HC_test, label='actual')
plt.plot(HC_crisis, label='crisis')
#plt.plot(HC_fc_series, label='forecast')
plt.plot(HC_fc_series2, label='forecast in crisis')

#plt.fill_between(HC_lower_series.index, HC_lower_series, HC_upper_series, 
 #                color='k', alpha=.15)
plt.fill_between(HC_lower_series2.index, HC_lower_series2, HC_upper_series2, 
                 color='k', alpha=.15)
plt.title('Forecast VS Actuals For Health Care')
plt.legend(loc='upper left', fontsize=8)
plt.show()
            


# In[42]:


# Calculate AIC for the predicted data and actual data

aic.aic(HC_fc_series, HC_test, 12)


# Test AIC

# In[43]:


def AIC(y,y_pre,k):
    n=len(y)
    resid = y - y_pre
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*np.log(float(SSR)/n)
    return AICValue


# In[44]:


AIC(FS_fc_series, FS_test, 17)


# In[45]:


MAPE(FS_fc_series, FS_test)


# In[ ]:


MAPE(HC_fc_series, HC_test)


# In[ ]:


AIC(HC_fc_series, HC_test, 12)


# Crisis MAPE

# In[ ]:


def MAPE(y,y_pre):
    y=np.array(y)
    y_pre=np.array(y_pre)
    n = len(y)
    return sum(np.abs((y - y_pre)/y))/n*100


# In[ ]:


MAPE(FS_fc_series2,FS_crisis)


# In[ ]:


MAPE(HC_fc_series2,HC_crisis)


# Crisis 欧氏距离

# In[51]:


def distance(hisA,hisB):
    d= np.sqrt(np.sum([(q-f)**2 for (q,f) in zip(hisA,hisB)]))
    return d


# In[52]:


distance(FS_fc_series2,FS_crisis)


# In[53]:


distance(HC_fc_series2,HC_crisis)


# Crisis chi-square

# In[48]:


def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


# In[49]:


chi2_distance(FS_fc_series2,FS_crisis)


# In[50]:


chi2_distance(HC_fc_series2,HC_crisis)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




