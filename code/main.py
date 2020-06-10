#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split


# ## Part 1 - Data Cleaning ##

# In[ ]:


### Merging the article and price datasets ###
df =pd.read_csv('article.csv')
df1 =pd.read_csv('price.csv')
df2=pd.read_csv('additional.csv')
df3=pd.read_csv('extra.csv')


# In[ ]:


new =pd.merge(left=df, right=df1, left_on="link", right_on="link")
new = new.drop_duplicates(subset="link", keep="first")
new =pd.merge(left=new, right=df2, left_on="link", right_on="link")
new =pd.merge(left=new, right=df3, left_on="link", right_on="link")
new.to_csv(r'C:\Users\user\Desktop\DM Project\merge.csv', index=False, header=True)
new.head()


# In[ ]:


missing_values = ["n/a", "na", "--", "nan", "-","NA", "NaN", " "]
df4 = pd.read_csv("merge.csv", na_values = missing_values)
print (df4.isnull().sum())


# In[ ]:


#Explore nulls
df4.columns
print(df4.isnull().sum())
#Fill nulls with Unknown 
df4['Colour'].fillna(value='Unknown', inplace=True)
print(df4.isnull().sum())


# In[ ]:


#Drop columns (useless/too many nulls)
df4.drop(columns=['version','link', 'Bore(mm)','Stroke(mm)','Compression Ratio'], inplace=True)
print(df4.isnull().sum())
#Drop Duplicates
df4.drop_duplicates(inplace=True)
print(df4.isnull().sum())


# In[ ]:


print(df4.info())


# In[ ]:


#Data transformation(price and engine capacity should be in float)
df4['Engine Capacity'] = df4['Engine Capacity'].replace('cc','',regex=True).astype(float)
df4['mileage'] = df4['mileage'].astype(float)
df4['Price'] = df4['Price'].replace('[RM,]', '',regex=True).astype(float)
df4['Seat Capacity'] = df4['Seat Capacity'].astype(float)
print(df4.info())


# In[ ]:


df5 = df4.copy()


# In[ ]:


#Imputate remaining missing values based on make
median = df5['Fuel Tank (litres)'].median()
df5['Fuel Tank (litres)'].fillna(median, inplace=True)
df5.isnull().sum()


# In[ ]:


#Drop nulls for remaining columns 
df5.dropna(inplace=True)
print(df5.isnull().sum())


# In[ ]:


dfnew = df5.copy()


# In[ ]:


dfnew = dfnew.rename(columns ={"mileage": "mileage_kms", "Price": "price",
                       "Engine Capacity": "engine_capacity", 
                       "Seat Capacity": "seat_capacity", "Colour": "colour",                                        
                       "Power(hp)": "power_hp", "Peak Torque(Nm)": "peak_torque_nm", 
                       "Length(mm)": "length", "Width(mm)": "width", "Height(mm)": "height",
                       "Fuel Tank (litres)": "fuel_tanks_litres"})
dfnew.head()


# In[ ]:


dfnew.describe()


# In[ ]:


dfnew.to_csv(r'C:\Users\user\Desktop\DM Project\clean_null.csv', index=False, header=True)


# ## Part 2 - Data Preprocessing & Exploratory Analysis ##

# #### Distribution Plots ####

# In[ ]:


# Plotting out distributions for the numeric variables to see whether they are normally distributed
warnings.filterwarnings("ignore")
sns.set()
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10))=plt.subplots(ncols=2,nrows=5,figsize=(12,20)) 
sns.distplot(dfnew['year'],ax=ax1);
sns.distplot(dfnew['mileage_kms'],ax=ax2);
sns.distplot(dfnew['price'],ax=ax3);            
sns.distplot(dfnew['engine_capacity'],ax=ax4);
sns.distplot(dfnew['power_hp'],ax=ax5);
sns.distplot(dfnew['peak_torque_nm'], ax=ax6);
sns.distplot(dfnew['fuel_tanks_litres'],ax=ax7);
sns.distplot(dfnew['length'],ax=ax8);
sns.distplot(dfnew['width'],ax=ax9);
sns.distplot(dfnew['height'],ax=ax10);


# In[ ]:


dfnew = dfnew[dfnew['price'].between(2700, 600000)] # Computing IQR
Q1 = dfnew['price'].quantile(0.25)
Q3 = dfnew['price'].quantile(0.75)
IQR = Q3 - Q1
# Filtering Values between Q1-1.5IQR and Q3+1.5IQR
dfnew = dfnew.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
dfnew.boxplot('price')


# In[ ]:


dfnew.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))


# #### Frequency Plots ####

# In[ ]:


categories = ['make','transmission', 'colour', 'seat_capacity'] 
# categorical features used in my analysis
ranges = [0,1,2,3,4] 
counts = []
x = []
y = []
for i,j in zip(ranges,categories):
    z = dfnew.groupby([j])['id'].count().sort_values(ascending=False).reset_index()
    counts.append(z)
    x.append(counts[i][j])
    y.append(counts[i]['id'])
    plt.figure(figsize=(10,4))
    sns.barplot(x[i],y[i])
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)


# #### Further Cleaning Process ####

# In[ ]:


#Cleaning mileage
dfnew = dfnew.drop(dfnew[dfnew.mileage_kms>300000].index)


# In[ ]:


#Cleaning make
print(dfnew.groupby(['make'])['id'].count().sort_values(ascending=True).reset_index())


# In[ ]:


# Drop make that less than 5
data = pd.read_csv('clean_null.csv', index_col='make')
data.drop(['McLaren','BMW Alpina','Bison', 'Cadillac', 'Fiat', 'Wald', 
           'Great Wall','Foton', 'Aston Martin', 'Shenyang Brilliance', 
           'Maxus', 'Alfa Romeo', 'Impul', 'Haval', 'Jeep', 'CAM'], inplace=True)
print(data.groupby(['make'])['id'].count().sort_values(ascending=True).reset_index())


# In[ ]:


data.to_csv(r'C:\Users\user\Desktop\DM Project\clean_data.csv', index=True, header=True)
data.head()


# #### Integrating more data - Countries ####

# In[ ]:


Malaysia = {'Proton', 'Perodua', 'Naza', 'Inokom'}
Germany = {'Audi','BMW','Mercedes-Benz','Skoda','Porsche','Smart','Volkswagen','Opel'}
Italy = {'Fiat','Iveco','Alfa', 'Ferrari', 'Lamborghini', 'Maserati'}
USA = {'Ford','Jeep','Chrysler','Chevrolet', 'Hummer'}
Japan = {'Honda','Nissan','Suzuki','Ssangyong','Hyundai','Mitsubishi','Mazda','Toyota','Kia','Subaru','Lexus', 'Isuzu', 'Daihatsu', 'Infiniti'}
Spain ={'Seat'}
Romania= {'Dacia'}
France={'Renault','Peugeot','Citroen','Ds'}
Sweden = {'Volvo','Saab'}
UK = {'Land Rover','Jaguar','MINI','Bentley','Rolls-Royce'}
China = {'Chery'}

mydata = {'Malaysia':Malaysia, 'Germany':Germany, 'Italy':Italy, 'USA':USA, "Japan":Japan, 'Spain':Spain, 'Romania':Romania, 'France':France, 'Sweden':Sweden, 'UK':UK, 'China':China}

mydicts = [{(z,i) for z in j} for i,j in mydata.items()]
mydict = {}
[mydict.update(i) for i in mydicts]
data = pd.read_csv('clean_data.csv')
data['country'] = data['make'].map(mydict)


# In[ ]:


data.to_csv(r'C:\Users\user\Desktop\DM Project\intc_data.csv', index=True, header=True)
data.head()


# ## Part 3 - Visualization ##

# In[ ]:


# Prepare columns for barplot
data = data.drop(data[data.price>600000].index)
yearly_price = data.groupby(['year'])['price'].mean().reset_index()
sem_price = data.groupby(['year'])['price'].sem().reset_index()


# In[ ]:


plt.figure(figsize=(10, 8));
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size":8});
sns.set(font_scale=1.1)
fig = sns.barplot(x=yearly_price['year'].astype(int),y= yearly_price['price'], yerr=sem_price['price'],capsize=4,errwidth=3,palette="Blues_d")
plt.ylabel('Price: RM',fontsize=16);
plt.xlabel('Model Year',fontsize=16);
plt.xticks(rotation=90)
plt.title('Average Used Car Prices Per Model Year',fontsize=22,fontweight='bold');
plt.savefig('pricect.png', dpi=400)


# In[ ]:


sem_price = data.groupby(['country'])['price'].sem().reset_index()
plt.figure(figsize=(10, 8));
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size":8});
sns.set(font_scale=1.1)
fig = sns.barplot(x=data['country'],y= data['price'],yerr=sem_price['price'],errwidth=3,palette="Blues_d")
plt.ylabel('Price: RM',fontsize=16);
plt.xlabel('Country',fontsize=16);
plt.xticks()
plt.title('Price per Country',fontsize=22,fontweight='bold');


# In[ ]:


sns.set(font_scale = 1.0)
plt.figure(figsize=(17, 10));
plot = sns.boxplot(x='mileage_kms',y='make',data=data,notch=True,orient='h',palette="coolwarm",showfliers=False)
plt.xlabel('Mileage',fontsize=17,fontweight="bold")
plt.ylabel('Car Maker',fontsize=17,fontweight="bold")
plt.title('Distribution of Mileage per Car Maker',fontsize=20,fontweight="bold")
plt.xlim()
plt.savefig('mileagect.png')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 10));
sns.boxplot(x='country',y='mileage_kms',data=data,notch=True,palette="coolwarm",showfliers=False)
plt.xlabel('Countries',fontsize=14,fontweight="bold")
plt.ylabel('Mileage',fontsize=14,fontweight="bold")
plt.title('Distribution of Mileage per Country of Manufacturer',fontsize=18,fontweight="bold")
plt.xlim()
plt.show()


# In[ ]:


sns.set(font_scale = 1.5)
plt.figure(figsize=(17, 10));
sns.boxplot(x='power_hp',y='make',data=data,notch=True,orient='h',palette='coolwarm',showfliers=False)
plt.xlabel('Horsepower',fontsize=17,fontweight="bold")
plt.ylabel('Car Maker',fontsize=17,fontweight="bold")
plt.title('Distribution of Horsepower per Car Maker',fontsize=22,fontweight="bold")
plt.xlim(0,500)
plt.show()


# In[ ]:


sns.set(style='dark')
plt.figure(figsize=(15, 10));
sns.boxplot(x='country',y='power_hp',data=data,showfliers=False,hue='country')
plt.xlabel('Countries',fontsize=16,fontweight="bold")
plt.ylabel('Horsepower',fontsize=16,fontweight="bold")
plt.title('Distribution of Horsepower per Country of Manufacturer',fontsize=18,fontweight="bold")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim()
plt.savefig('hpct.png',dpi=400)
plt.show()


# In[ ]:


sns.set(font_scale = 1.5)
plt.figure(figsize=(17, 10));
sns.boxplot(x='fuel_tanks_litres',y='make',data=data,notch=True,orient='h',palette='coolwarm',showfliers=False)
plt.xlabel('Fuel Tank Size',fontsize=17,fontweight="bold")
plt.ylabel('Car Maker',fontsize=17,fontweight="bold")
plt.title('Distribution of Fuel Tank Size per Car Maker',fontsize=22,fontweight="bold")
plt.xlim(0,120)
plt.show()


# In[ ]:


sns.set(style='dark')
plt.figure(figsize=(15, 10));
sns.boxplot(x='country',y='fuel_tanks_litres',data=data,showfliers=False,hue='country')
plt.xlabel('Countries',fontsize=16,fontweight="bold")
plt.ylabel('Fuel Tank (L)',fontsize=16,fontweight="bold")
plt.title('Distribution of Fuel Tank Size per Country of Manufacturer',fontsize=18,fontweight="bold")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim()
plt.savefig('ftct.png',dpi=400)
plt.show()


# In[ ]:


from PIL import Image
#Generate frequencies using counter
freqs = Counter(data['make'])
df = pd.DataFrame.from_dict(freqs, orient='index').reset_index()
df.columns = ['brands','freqs']
# Generate a word cloud with freqs
wc = WordCloud(background_color="white", max_words=1000, contour_width=2,contour_color='black')
wc.generate_from_frequencies(freqs)
plt.figure(figsize=(10, 8))
plt.axis("off")
make = data.groupby(['make'])['id'].count().sort_values(ascending=False).reset_index()
plt.savefig('wordcloudcar.png')
plt.imshow(wc, interpolation='bilinear');


# In[ ]:


sns.set(font_scale = 1.25)
plt.figure(figsize=(10, 9));
sns.set_style("ticks",{"xtick.major.size": 12, "ytick.major.size":8})
make = data.groupby(['make'])['id'].count().sort_values(ascending=False).reset_index();
sns.barplot(make['id'],make['make'],orient='h',color="#2874A6");
plt.title("Number of Cars Per Car Maker",fontsize=22)
plt.ylabel('Car Maker',fontsize=16.5)
plt.xlabel('Number of Cars',fontsize=16.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=13)
plt.savefig('numberofcars.png')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
numeric_corr = data[['price','power_hp','year','mileage_kms','bore','stroke', 'peak_torque_nm','fuel_tanks_litres','length','width','height']].corr(method='spearman');
sns.heatmap(numeric_corr,annot=True, ax=ax);
plt.title('Used Car Dataset');


# ## Part 4 - Machine Learning Models ##

# In[ ]:


df = pd.read_csv('intc_data.csv')
df.drop(columns=['Unnamed: 0','model', 'id'], inplace=True)
df.head()


# #### Handling Categorical Data (One Hot Encoding) ####

# In[ ]:


X = df[['year', 'mileage_kms', 'power_hp', 'transmission', 'make']]
Y = df.price
X = pd.get_dummies(data=X)


# In[ ]:


X.head()


# #### Data Splitting ####

# In[ ]:


# Splitting data into training and testing.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)


# ### Linear Regression ###

# In[ ]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='purple')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='purple')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)


# In[ ]:


print('Variance score: %.2f' % r2_score(Y_test, predicted))


# ### KNN Regression ###

# In[ ]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6) # value changed based on histogram with the lowest RMSE.
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)


# In[ ]:


from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# In[ ]:


rmse_l = []
num = []
for n in range(2, 16):
    knn = neighbors.KNeighborsRegressor(n_neighbors=n)
    knn.fit(X_train, Y_train)
    predicted = knn.predict(X_test)
    rmse_l.append(np.sqrt(mean_squared_error(Y_test, predicted)))
    num.append(n)


# In[ ]:


df_plt = pd.DataFrame()
df_plt['rmse'] = rmse_l
df_plt['n_neighbors'] = num
ax = plt.figure(figsize=(7,5))
sns.barplot(data = df_plt, x = 'n_neighbors', y = 'rmse')
plt.show()


# ### Gradient Boosting Regresssion ###

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

r_sq = []
deep = []
mean_scores = []

for n in range(3, 11):
    gbr = GradientBoostingRegressor(loss ='ls', max_depth=n)
    gbr.fit (X, Y)
    deep.append(n)
    r_sq.append(gbr.score(X, Y))
    mean_scores.append(cross_val_score(gbr, X, Y, cv=12).mean())


# In[ ]:


plt_gbr = pd.DataFrame()

plt_gbr['mean_scores'] = mean_scores
plt_gbr['depth'] = deep
plt_gbr['R²'] = r_sq

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='R²')
plt.show()

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='mean_scores')
plt.show()


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='red')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='red')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

rmse = np.sqrt(mean_squared_error(Y_test, predicted))
scores = cross_val_score(gbr, X, Y, cv=12)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)


# In[ ]:


print('Variance score: %.2f' % r2_score(Y_test, predicted))


# #### Model Comparison ####

# <table class="table table-bordered">
#     <thead>
#       <tr>
#         <th>Model</th>
#         <th>Variance Score</th>
#         <th>RMSE</th>
#       </tr>
#     </thead>
#     <tbody>
#       <tr>
#         <td>Multiple Linear Regression</td>
#         <td>0.80</td>
#         <td>47455.31</td>
#       </tr>
#       <tr>
#         <td>KNN</td>
#         <td>0.77</td>
#         <td>51296.55</td>
#       </tr>
#       <tr style="color: green">
#         <td><b>Gradient Boosting</b></td>
#         <td><b>0.91</b></td>
#         <td><b>32238.23</b></td>
#       </tr>
#     </tbody>
# </table>

# #### Prediction with best model (Gradient Boosting) ####

# In[ ]:


user_input = {'year':2010, 'mileage_kms':82499.5, 'power_hp':109.0, 'transmission':'Automatic', 'make':'Toyota'}
def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(45)
    # set the numerical input as they are
    enc_input[0] = data['year']
    enc_input[1] = data['mileage_kms']
    enc_input[2] = data['power_hp']
    ##################### Make #########################
    # get the array of make categories
    make = df.make.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'make_'+data['make']
    # search for the index in columns name list 
    make_column_index = X.columns.tolist().index(redefinded_user_input)
    #print(mark_column_index)
    enc_input[make_column_index] = 1
    ##################### Transmission ####################
    # get the array of transmission
    transmission = df.transmission.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'transmission_'+data['transmission']
    # search for the index in columns name list 
    transmission_column_index = X.columns.tolist().index(redefinded_user_input)
    enc_input[transmission_column_index] = 1
    return enc_input


# In[ ]:


print(input_to_one_hot(user_input))


# In[ ]:


a = input_to_one_hot(user_input)


# In[ ]:


price_pred = gbr.predict([a])


# In[ ]:


price_pred[0]

