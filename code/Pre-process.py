#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


### Merging the article and price datasets ###


# In[2]:


df=pd.read_csv('pricedata.csv')
df1=pd.read_csv('article_data.csv')


# In[ ]:


df2=pd.merge(left=df1, right=df, left_on="link", right_on="link")
df2


# In[ ]:


set1 = df2.drop_duplicates(subset='link', keep="first")
set1


# In[ ]:


set1.to_csv(r'C:\Users\user\Desktop\New folder\merge_data.csv', index=False, header=True)
set1


# In[ ]:


### Merging with the additional data ###


# In[ ]:


df=pd.read_csv('merge_data.csv')
df1=pd.read_csv('additional.csv')


# In[ ]:


df2=pd.merge(left=df, right=df1, left_on="link", right_on="link")


# In[ ]:


df2.to_csv(r'C:\Users\user\Desktop\New folder\rawdata.csv', index=False, header=True)


# In[ ]:


### Data Cleaning ###


# In[3]:


data=pd.read_csv('final.csv')


# In[4]:


data["Seat Capacity"].unique()


# In[5]:


data["Colour"].unique()


# In[6]:


data = data.replace("-",np.nan)


# In[7]:


data.isnull().sum()


# In[8]:


df3 = data.dropna(subset=['Price', 'Engine Capacity', 'Seat Capacity'])
df3


# In[9]:


df3.isnull().sum()


# In[11]:


df3.info()


# In[12]:


df_new=df3.copy()


# In[18]:


df_new['Engine Capacity'] = df_new['Engine Capacity'].replace('cc','',regex=True)
df_new['Engine Capacity']=pd.to_numeric(df_new['Engine Capacity'],downcast='integer',errors='coerce')
df_new['Price'] = df_new['Price'].replace('[RM,]', '',regex=True).astype(float)
df_new.head()


# In[19]:


df4 = df_new.drop(['id'], axis=1)
df4


# In[20]:


df4.describe()


# In[21]:


df4.corr()


# In[22]:


print(df4.Price.mean())
print(df4.Price.median())


# In[12]:


print(df1.Price.max())
print(df1.Price.min())


# In[13]:


x = df1.Price
plt.figure(figsize=(10,6))
sns.distplot(x).set_title('Frequency Distribution Plot of Prices')


# In[23]:


df4.to_csv(r'C:\Users\user\Desktop\New folder\final.csv', index=False, header=True)
df4

