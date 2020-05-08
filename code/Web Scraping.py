#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from time import sleep
from random import randint
import requests 
from bs4 import BeautifulSoup as soup
import warnings
warnings.warn("Warning Simulation")
from IPython.core.display import clear_output
import time 
start_time = time.time()


# In[13]:


url_test = 'https://www.carlist.my/used-cars-for-sale/malaysia'
page = requests.get(url_test)
print(page)


# In[48]:


requests = 0
pages = [str(i) for i in range(0,1)]
print(pages)

cars = []
price = []
years = []
mileages = []
transmissions =[]
makes = []
models =[]
listing_ids = []
links = []


# In[64]:


pages=0

for pages in range (0,1000):

        # Make a get request
       
        newpage = requests.get('https://www.carlist.my/used-cars-for-sale/malaysia?page_number='+str(pages)+'&page_size=25')
                 
            
        # Pause the loop
        sleep(randint(8,15))

        # Monitor the requests
        requests += 1
        elapsed_time = time.time() - start_time
        print('Request:{}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))
        clear_output(wait = True)

        # Throw a warning for non-200 status codes
        if newpage.status_code != 200:
            warn('Request: {}; Status code: {}'.format(requests, newpage.status_code))
        
        # Parse the content of the request with BeautifulSoup
     
        soup = BeautifulSoup(newpage.content, 'html.parser')

        car_list = soup.find(id="classified-listings-result")

        # For every car addtional information
        car = [c["data-display-title"] for c in car_list.select("article")]
        cars.extend (car)
              
        year = [y["data-year"] for y in car_list.select("article")]
        years.extend(year)
        
        mileage = [m["data-mileage"] for m in car_list.select("article")]
        mileages.extend(mileage)
        
        transmission = [t["data-transmission"] for t in car_list.select("article")]
        transmissions.extend(transmission)
        
        make = [mk["data-make"] for mk in car_list.select("article")]
        makes.extend(make)
        
        model = [md["data-model"] for md in car_list.select("article")]
        models.extend (model)     
             
        listing_id = [lis["data-listing-id"] for lis in car_list.select("article")]
        listing_ids.extend(listing_id)
        
        link = [url["data-url"] for url in car_list.select("article")]
        links.extend (link)

car_list.head


# In[65]:


print(len(models))
print(len(listing_ids))
print(len(makes))
print(len(years))
print(len(mileages))
print(len(transmissions))
print(len(links))


# In[67]:


used_cars = pd.DataFrame({'id': listing_ids,
                          'model': models, 
                          'make' : makes , 
                          'version' : cars, 
                          'year': years, 
                          'mileage': mileages, 
                          'transmission': transmissions, 
                          'link': links})
print(used_cars.info())
used_cars


# In[68]:


df1=pd.DataFrame(used_cars)
df1.to_csv(r'C:\Users\user\Desktop\New folder\article_data.csv', index=False, header=True)
df1


# In[ ]:


pages=0

for pages in range (0,1):
    pages+=1
    url = 'https://www.carlist.my/used-cars-for-sale/malaysia?page_number='+str(pages)+'&page_size=25'
    page2 = requests.get(url)
    soup2 = soup(page2.content, 'html.parser')
    user_data = soup2.find_all('div', attrs={'class' : 'grid grid--full cf'})
    

print(user_data[24].prettify())


# In[ ]:


car_list = []
pages=0
for pages in range (0,1000):
    pages+=1
    url = 'https://www.carlist.my/used-cars-for-sale/malaysia?page_number='+str(pages)+'&page_size=25'
    page2 = requests.get(url)
    soup2 = soup(page2.content, 'html.parser')
    user_data = soup2.find_all('div', attrs={'class' : 'grid grid--full cf'})
    if user_data!=[]:
        for item in user_data:
            d={}            
                     
            try:
                d['Price'] = item.find('div',{'class' : 'listing__price delta weight--bold'}).text
            except:
                d['Price']= "NA" 
            
            try:
                d['link'] = item.find('a', href=True).get('href')
            except:
                d['link']= "NA"   
            
            car_list.append(d)


# In[ ]:


df=pd.DataFrame(car_list)
df.to_csv(r'C:\Users\user\Desktop\New folder\pricedata.csv', index=False, header=True)
df


# In[43]:


df=pd.read_csv('pricedata.csv')
df


# In[26]:


url_test = 'https://www.carlist.my/used-cars-for-sale/malaysia'
page = requests.get(url_test)
print(page)


# In[27]:


additional=[]
allref= df['link']

for ref in allref:
    page3=requests.get(ref+'#spec')
    soup3=soup(page3.content,'html.parser')
    car_data=soup3.find_all('div', class_='listing__key-listing__list')
    if car_data!=[]:
        for items in car_data:
            e={}
            e['link']=ref
            
            try:
                e['Engine Capacity']=items.find('span', text='Engine Capacity').find_next('span').text
            except:
                e['Engine Capacity']="NA"
            try:
                e['Seat Capacity']=items.find('span', text='Seat Capacity').find_next('span').text
            except:
                e['Seat Capacity']="NA"
            try:
                e['Colour']=items.find('span', text='Colour').find_next('span').text
            except:
                e['Colour']="NA"

    additional.append(e)


# In[ ]:


df1=pd.DataFrame(additional)
df1


# In[ ]:


df1= df1.drop_duplicates(subset='link', keep="first")


# In[29]:


df1.to_csv(r'C:\Users\user\Desktop\New folder\additional.csv', index=False, header=True)

