#!/usr/bin/env python
# coding: utf-8

# In[2]:



from sklearn import metrics
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


# In[43]:


def MAPE(y,y_pre):
    y=np.array(y)
    y_pre=np.array(y_pre)
    n = len(y)
    return sum(np.abs((y - y_pre)/y))/n*100


# In[4]:


def AIC(y,y_pre,k):
    n=len(y)
    resid = y - y_pre
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*np.log(float(SSR)/n)
    return AICValue


# In[5]:


FF=pd.read_excel('D:\\FF_Model_Test1.xlsx')
CAPM=pd.read_excel('D:\\CAPM_Test.xlsx')
FFcrisis=pd.read_excel('D:\\FF_Model_Forecasting.xlsx')
CAPMcrisis=pd.read_excel('D:\\CAPM_Forecasting.xlsx')


# In[6]:


CAPMcrisis


# In[ ]:





# In[ ]:





# MAPE

# In[31]:


MAPE(FF['Testing DJUSHC'],FF['DJUSHC'])


# In[32]:


MAPE(FF['Testing DJUSGF'],FF['DJUSGF'])


# In[44]:


MAPE(CAPM['Testing DJUSHC'],CAPM['DJUSHC'])


# In[45]:


MAPE(CAPM['Testing DJUSGF'],CAPM['DJUSGF'])


# In[46]:



MAPE(FFcrisis['Forecasted DJUSGF'],FFcrisis['DJUSGF in crisis'])


# In[47]:


MAPE(FFcrisis['Forecasted DJUSHC'],FFcrisis['DJUSHC in crisis'])


# In[48]:


MAPE(CAPMcrisis['Forecasted DJUSGF'],CAPMcrisis['DJUSGF in crisis'])


# In[49]:



MAPE(CAPMcrisis['Forecasted DJUSHC'],CAPMcrisis['DJUSHC in crisis'])


# RMSE

# In[11]:



np.sqrt(metrics.mean_squared_error(FFcrisis['Forecasted DJUSGF'],FFcrisis['DJUSGF in crisis']))


# In[12]:



np.sqrt(metrics.mean_squared_error(FFcrisis['Forecasted DJUSHC'],FFcrisis['DJUSHC in crisis']))


# In[13]:


np.sqrt(metrics.mean_squared_error(CAPMcrisis['Forecasted DJUSGF'],CAPMcrisis['DJUSGF in crisis']))


# In[14]:


np.sqrt(metrics.mean_squared_error(CAPMcrisis['Forecasted DJUSHC'],CAPMcrisis['DJUSHC in crisis']))


# AIC

# In[30]:



from RegscorePy import *
aic.aic(FF['Testing DJUSHC'],FF['DJUSHC'],6)


# In[16]:


aic.aic(FF['Testing DJUSGF'],FF['DJUSGF'],7)


# In[36]:


aic.aic(CAPM['Testing DJUSHC'],CAPM['DJUSHC'],4)


# In[37]:


aic.aic(CAPM['Testing DJUSGF'],CAPM['DJUSGF'],5)


# In[19]:



def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


# In[20]:


chi2_distance(FFcrisis['Forecasted DJUSGF'],FFcrisis['DJUSGF in crisis'])


# In[21]:



chi2_distance(FFcrisis['Forecasted DJUSHC'],FFcrisis['DJUSHC in crisis'])


# In[22]:


chi2_distance(CAPMcrisis['Forecasted DJUSGF'],CAPMcrisis['DJUSGF in crisis'])


# In[23]:




chi2_distance(CAPMcrisis['Forecasted DJUSHC'],CAPMcrisis['DJUSHC in crisis'])


# In[24]:


def distance(hisA,hisB):
    d= np.sqrt(np.sum([(q-f)**2 for (q,f) in zip(hisA,hisB)]))
    return d


# In[25]:


distance(FFcrisis['Forecasted DJUSGF'],FFcrisis['DJUSGF in crisis'])


# In[26]:


distance(FFcrisis['Forecasted DJUSHC'],FFcrisis['DJUSHC in crisis'])


# In[27]:


distance(CAPMcrisis['Forecasted DJUSGF'],CAPMcrisis['DJUSGF in crisis'])


# In[28]:




distance(CAPMcrisis['Forecasted DJUSHC'],CAPMcrisis['DJUSHC in crisis'])


# In[ ]:





# In[ ]:




