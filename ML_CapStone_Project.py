#!/usr/bin/env python
# coding: utf-8

# # Customer Behaviour Analysis with Machine Learning

# # Scenario 1

# # Task-1

# 1.Importing the relevant packages (Packages example: numpy, pandas....)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans
import joblib
import streamlit as st
from numpy import outer


# 2.Extract the dataset from the file named "dataset.csv" and save it in customer variable

# In[2]:


customer=pd.read_csv('Customerdata.csv')


# 3.Is the dataset successfully called? Can you check with the top 10 records?

# In[3]:


customer.head(10)


# 4.Check for the structure and dimensions of the dataset

# In[4]:


customer.shape


# In[5]:


customer.ndim


# In[6]:


customer.size

5.Show the column names
# In[7]:


customer.columns


# # Task 2:

# 1.Create a new column "Age" by subtracting the column Year\_Birth from 2015

# In[8]:


customer['Age']=2015-customer['Year_Birth']


# In[9]:


customer.Age.head(1)


# 2.Check the statistics of the dataset

# In[10]:


customer.describe()


# 3.Work on analysing Missing values and show the output

# In[11]:


customer.isnull().sum()


# 4.Display the missing values using heatmap and remove the y labels while plotting.

# In[12]:


plt.figure(figsize=(15,5))
sns.heatmap(customer.isnull(), cbar=False,yticklabels=False)


# 5.Drop the missing values

# In[13]:


customer.isnull().sum()


# In[14]:


customer['Income'].dropna(inplace=True)


# In[15]:


customer.shape


# # Scenario 2

# Congratulations on getting the dataset in workable format. You will be learning exploratory data analysis here that can assist us to further clean data. 
# 
# We have dataset with 2240 records and 30 columns.

# In[16]:


customer.shape


# In[17]:


customer.dropna(inplace=True)


# # Task

# 1. From the enrolment date of customers, let's calculate the number of months the customers are affiliated with the company with name "Month\_Customer". 

# In[18]:


customer.head(3)


# In[19]:


customer['year'] = pd.DatetimeIndex(customer['Dt_Customer']).year
customer['month'] = pd.DatetimeIndex(customer['Dt_Customer']).month


# In[20]:


customer['Month_Customer']=12*(2015-customer['year'])+(customer['month']-1)


# In[21]:


#check
#customer['month'].min()


# In[22]:


customer['Month_Customer'].head(2)


# In[23]:


customer=customer.drop(['year','month'],axis=1)


# 2. Create a column named as "TotalSpendings". This is sum of amount spent on products.

# In[24]:


#'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds',
customer['TotalSpendings']=customer['MntWines']+customer['MntFruits']+customer['MntMeatProducts']+customer['MntSweetProducts']+customer['MntFishProducts']+customer['MntGoldProds']


# In[25]:


customer['TotalSpendings'].head(2)


# 3. On the basis of Age let's divide the customers into different age groups and create a column "AgeGroup". The logic for that is;
# 
#         
# 
#         Age Group is Teen for age less than 19
# 
#         Age Group is Adults for age between 20 and 39
# 
#         Age Group is Middle Age Adults for age between 40 and 59
# 
#         Age Group is Senior for age more than 60

# In[26]:


customer.loc[(customer.Age <=19),'AgeGroup']='Teen'
customer.loc[(customer.Age >=20) & (customer.Age <=39),'AgeGroup']='Adults'
customer.loc[(customer.Age >=40) & (customer.Age <=59),'AgeGroup']='Middle_Age_Adults'
customer.loc[(customer.Age >=60),'AgeGroup']='Senior'


# In[27]:


customer["AgeGroup"].isnull().sum()


# 4. Information is given separately for kids and teens at home for every customers. Let's sum them up, as they can be better represented together as the number of children at home, with column name "Children".

# In[28]:


customer['Children']=customer['Kidhome']+customer['Teenhome']


# In[29]:


customer['Children'].head(1)


# In[30]:


customer.columns


# In[31]:


customer.Marital_Status


# In[32]:


customer['Marital_Status']=customer['Marital_Status'].replace(to_replace=['Together'],value='Married')
customer['Marital_Status']=customer['Marital_Status'].replace(to_replace=['Divorced','Widow','Alone','Absurd','YOLO'],value='Single')


# In[33]:


customer['Marital_Status'].unique()


# 6. There seems to be some outliers in the Age and Income columns. Let's check them. Use boxplot for each individual columns. Update the dataset by removing the records that are outliers.

# In[34]:


#old
customer['Age'].plot(kind = 'box',figsize=(11,3))


# In[35]:


#old
customer['Income'].plot(kind = 'box',figsize=(11,3))


# In[36]:


customer['Age'].describe()


# In[37]:


q3 = 56 #75%
q1 = 38 #25%
IQR_ol = q3 - q1
val = q3 + (1.5 * IQR_ol) #q1-(1.5*IQR)
val


# In[38]:



customer=customer.drop(customer[customer['Age'] > 83].index, inplace = False)


# In[39]:


#new after removing outliers
customer['Age'].plot(kind = 'box',figsize=(11,3))


# In[40]:


customer['Income'].describe()


# In[41]:


q3 = 68522 #75%
q1 = 35303 #25%
IQR_ol = q3 - q1
val = q3 + (1.5 * IQR_ol) #q1-(1.5*IQR)
val


# In[42]:


customer=customer.drop(customer[customer['Income'] > 118350.5].index, inplace = False)


# In[43]:


#new after removing outliers
customer['Income'].plot(kind = 'box',figsize=(11,3))


# # Scenario 3

# So far you have completed tasks such as extracting the dataset, creating new columns, cleaning the dataset and removing the outliers. 
# 
# We have dataset with 2205 records and 34 columns.

# In[44]:


customer.shape


# # Task
# EDA

# In[97]:


customer.describe()


# In[101]:


customer["Age"].skew()


# In[106]:


customer["Age"].kurt()


# Categorical Variable
MaritalStatus
# In[48]:


maritalstatus = customer.Marital_Status.value_counts()

fig = px.pie(maritalstatus, 
             values = maritalstatus.values, 
             names = maritalstatus.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 4)))
fig.show() 


# 2/3 of customers live with a partner while around 1/3 are single
# Average Spendings: Marital Status Wise
# In[49]:


maritalspending = customer.groupby('Marital_Status')['TotalSpendings'].mean().sort_values(ascending=False)
maritalspending_df = pd.DataFrame(list(maritalspending.items()), columns=['Marital Status', 'Average Spending'])

plt.figure(figsize=(13,5))
sns.barplot(data = maritalspending_df, x="Average Spending", y="Marital Status", palette='rocket');

plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Average Spending', fontsize=13, labelpad=13)
plt.ylabel('Marital Status', fontsize=13, labelpad=13);


# In[50]:


sns.boxplot(x="Marital_Status", y="TotalSpendings", data=customer, palette='rocket')


# Although in the minority, singles spend more on average than customers with partners.
# Education Level
# In[51]:


education = customer.Education.value_counts()

fig = px.pie(education, 
             values = education.values, 
             names = education.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# Half of the customers are university graduates
# There are more customers who have a PhD degree than customers who have a Master's degree
# Child Status
# In[52]:


children = customer.Children.value_counts()

fig = px.pie(children, 
             values = children.values, 
             names = children.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', 
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# About 50% of customers have only one child
#     28% of customers do not have children at home while 19% of them have 2 children

# Numerical Variable
Average Spendings: Child Status Wise
# In[53]:


childrenspending = customer.groupby('Children')['TotalSpendings'].mean().sort_values(ascending=False)
childrenspending_df = pd.DataFrame(list(childrenspending.items()), columns=['No. of Children', 'Average Spending'])

plt.figure(figsize=(10,5))

sns.barplot(data=childrenspending_df,  x="No. of Children", y="Average Spending", palette='rocket_r');
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('No. of Children', fontsize=13, labelpad=13)
plt.ylabel('Average Spending', fontsize=13, labelpad=13);


# Customers who do not have children at home spend more than customers who have 1 child.
# Customers who have 1 child have higher production than customers who have 2 and 3 children.
# 
# 
Age Distribution of Customers
# In[54]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = customer.Age, color='salmon')
ax.set(title = "Age Distribution of Customers");
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Age ', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);


# The age of the customers is almost normally distributed, with the majority of customers between 40 and 60 years old.
Relationship: Age vs Spendings
# In[55]:




plt.figure(figsize=(20,10))
sns.scatterplot(x=customer.Age, y=customer.TotalSpendings, s=100, color ='black');

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Age', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);


# There seems to be no clear relationship between the age of customers and their shopping habits.
Customers Segmentation: Age Group Wise
# In[56]:




agegroup = customer.AgeGroup.value_counts()

fig = px.pie(labels = agegroup.index, values = agegroup.values, names = agegroup.index, width = 550, height = 550)

fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.4, 
                  marker = dict(colors = ['#3D0C02', '#800000'  , '#C11B17','#C0C0C0'], 
                                line = dict(color = 'white', width = 2)))

fig.update_layout(annotations = [dict(text = 'Age Groups', 
                                      x = 0.5, y = 0.5, font_size = 20, showarrow = False,                                       
                                      font_color = 'black')],
                  showlegend = False)

fig.show()


# More than 50% of customers are Middle Age Adults between 40 and 60
# The 2nd most popular age category is Adults, aged between 20 and 40
Average Spendings: Age Group Wise
# In[57]:


agegroupspending = customer.groupby('AgeGroup')['TotalSpendings'].mean().sort_values(ascending=False)
agegroupspending_df = pd.DataFrame(list(agegroup.items()), columns=['Age Group', 'Average Spending'])

plt.figure(figsize=(20,10))

sns.barplot(data = agegroupspending_df, x="Average Spending", y='Age Group', palette='rocket_r');
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Average Spending', fontsize=20, labelpad=20)
plt.ylabel('Age Group', fontsize=20, labelpad=20);


# Middle-aged adults spend more than other age groups.
Income Distribution of Customers
# In[58]:


plt.figure(figsize=(10,5))
ax = sns.histplot(data = customer.Income, color = "indianred")
ax.set(title = "Income Distribution of Customers");

plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.xlabel('Income', fontsize=13, labelpad=13)
plt.ylabel('Counts', fontsize=13, labelpad=13);

Relationship: Income vs Spendings
# In[59]:




plt.figure(figsize=(20,10))


sns.scatterplot(x=customer.Income, y=customer.TotalSpendings, s=100, color='black');

plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);


# The relationship is linear. Customers with higher salaries spend more
Most Bought Products
# In[60]:




products = customer[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
product_means = products.mean(axis=0).sort_values(ascending=False)
product_means_df = pd.DataFrame(list(product_means.items()), columns=['Product', 'Average Spending'])

plt.figure(figsize=(15,10))
plt.title('Average Spending on Products')
sns.barplot(data=product_means_df, x='Product', y='Average Spending', palette='rocket_r');
plt.xlabel('Product', fontsize=20, labelpad=20)
plt.ylabel('Average Spending', fontsize=20, labelpad=20);

Wine and Meat products are the most popular products among customers
Candy and Fruit are not often bought
# 4.Check distribution of variables

# In[ ]:


customer.hist(figsize = (20,20), bins = 50)
plt.show()


# # Scenario 4

# In[62]:


X = customer.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits','MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Dt_Customer', 'Z_CostContact',
                          'Z_Revenue', 'Recency', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                          'AcceptedCmp1', 'AcceptedCmp2', 'Complain',  'Response', 'AgeGroup'], axis=1)


# In[63]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


options = range(2,9)
inertias = []

for n_clusters in options:
    model = KMeans(n_clusters, random_state=42).fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(20,10))    
plt.title("No. of clusters vs. Inertia")
plt.plot(options, inertias, '-o', color = 'black')
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('No. of Clusters (K)', fontsize=20, labelpad=20)
plt.ylabel('Inertia', fontsize=20, labelpad=20);


# In[64]:


model = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(X)


joblib.dump(model,'customer.pkl')

#Predicting value on the test set

mod = joblib.load('customer.pkl')
preds = model.predict(X)


customer_kmeans = X.copy()
customer_kmeans['clusters'] = preds
X['clusters']=preds


# In[65]:


#Income
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Income',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Income', fontsize=50, labelpad=20);


# In[66]:




#Total Spending
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'TotalSpendings',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=50, labelpad=20);


# In[67]:




#Month Since Customer
plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Month_Customer',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Month Since Customer', fontsize=50, labelpad=20);


# In[68]:


plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Age',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Age', fontsize=50, labelpad=20);


# In[69]:




plt.figure(figsize=(20,10))

sns.boxplot(data=customer_kmeans, x='clusters', y = 'Children',palette='rocket_r');
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('No. of Children', fontsize=50, labelpad=20);


# In[70]:


customer_kmeans.clusters = customer_kmeans.clusters.replace({1: 'A',
                                                             2: 'B',
                                                             3: 'C',
                                                             0: 'D'})

customer['clusters'] = customer_kmeans.clusters


# In[71]:


cluster_counts = customer.clusters.value_counts()

fig = px.pie(cluster_counts, 
             values = cluster_counts.values, 
             names = cluster_counts.index,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20,
                  marker = dict(line = dict(color = 'white', width = 2)))
fig.show()


# In[72]:




plt.figure(figsize=(20,10))
sns.scatterplot(data=customer, x='Income', y='TotalSpendings', hue='clusters', palette='rocket_r');
plt.xlabel('Income', fontsize=20, labelpad=20)
plt.ylabel('Total Spendings', fontsize=20, labelpad=20);


# 4 clusters can be easily identified from the plot above
# Those who earn more also spend more
Spending Habits by Clusters
# In[73]:


cluster_spendings = customer.groupby('clusters')[['MntWines', 'MntFruits','MntMeatProducts', 
                                                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()

cluster_spendings.plot(kind='bar', stacked=True, figsize=(9,7), color=['#dc4c4c','#e17070','#157394','#589cb4','#bcb4ac','#3c444c'])

plt.title('Spending Habits by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Spendings', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');

Purchasing Habits by Clusters
# In[74]:




cluster_purchases = customer.groupby('clusters')[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                                                  'NumStorePurchases', 'NumWebVisitsMonth']].sum()

cluster_purchases.plot(kind='bar', color=['#dc4c4c','#157394','#589cb4','#bcb4ac','#3c444c'], figsize=(9,7))

plt.title('Purchasing Habits by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Purchases', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');


# 
# Promotions Acceptance by Clusters

# In[75]:




cluster_campaign = customer.groupby('clusters')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 
                                                  'AcceptedCmp5', 'Response']].sum()

plt.figure(figsize=(30,15))
cluster_campaign.plot(kind='bar', color=['#dc4c4c','#e17070','#157394','#589cb4','#bcb4ac','#3c444c'],figsize=(9,7))

plt.title('Promotions Acceptance by Cluster')
plt.xlabel('Clusters', fontsize=20, labelpad=20)
plt.ylabel('Promotion Counts', fontsize=20, labelpad=20);
plt.xticks(rotation=0, ha='center');


# In[76]:


customer.head()


# In[77]:


X['clusters'].replace({1: 'A',2: 'B',3: 'C',0: 'D'},inplace=True)


# In[78]:


customer_kmeans=X.copy()


# Create a new dataset "customer\_kmeans" that has all columns of "X" and also include predicted labels.

# In[79]:


customer_kmeans.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




