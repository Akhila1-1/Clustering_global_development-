#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df= pd.read_csv("C:\\Users\\maheh\\Downloads\\World_development_mesurement (1).csv")
df


# In[3]:


df.shape


# In[4]:



df.columns


# In[5]:



df.info()


# In[6]:


df.describe()


# In[7]:


#Before imputation
df.isnull().sum()


# In[8]:


#finding features with null values
[features for features in df.columns if df[features].isnull().sum()>0]


# In[9]:


#Heatmap to show null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[3]:


#Imputation
for columns in df.columns:
    if df[columns].dtype == 'int':
        df[columns].fillna(df[columns].mean(),inplace=True)
    if df[columns].dtype == 'float':
        df[columns].fillna(df[columns].mean(),inplace=True)
    else:
        df[columns].fillna(df[columns].mode()[0],inplace=True)
df


# In[11]:


#After Imputation
df.isnull().sum()


# In[4]:


# Function to clean individual text values
import re
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[$%]', '', text)  # Remove dollar and percent signs
        text = re.sub(r'[^\d.]', '', text)  # Remove all non-digit characters except the decimal point
        text = text.strip()  # Strip any leading/trailing whitespace
    return text

# Columns to clean
columns_to_clean = ['GDP', 'Health Exp/Capita', 'Business Tax Rate', 'Tourism Inbound', 'Tourism Outbound']

# Loop through each column and apply the cleaning process
for column in columns_to_clean:
    df[column] = df[column].astype(str)  # Ensure the column is treated as a string
    df[column] = df[column].apply(clean_text)  # Apply the cleaning function
    df[column] = df[column].replace('', '0')  # Replace empty strings with 0 to avoid conversion errors
    df[column] = df[column].astype(float)  # Convert the cleaned text to float

# Verify the cleaned DataFrame
df


# In[13]:


#Finding the correalation between the features
# Calculate the correlation matrix 
correlation_matrix = df.corr() 
# Create a heatmap 
plt.figure(figsize=(12, 8)) 
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap') plt.show()


# In[14]:


#dropping feature No.of records feature as it'st.deviation is zero
#df.drop(columns=["Number of Records"],inplace=True)
df.drop(columns=["Country"],inplace=True)


# In[15]:


#df["Country"].value_counts().index


# In[16]:


df.hist(figsize=(20,20)) #univariate analysis


# In[17]:


skewness_values=df.skew()
skewness_values


# In[5]:


# Loop through the skewness values and apply log transformation to features with positive skew > 1
skewness_values = df.skew()
for feature, skewness in skewness_values.items():
    if skewness > 1:  # Apply log transformation to highly positively skewed features
        df[feature] = np.log(df[feature] + 1)  # Add 1 to avoid issues with zero values


# In[6]:


#After log-transformation

df.hist(figsize=(20,20))


# In[20]:


#Boxplot to visualize outliers (univariate analysis)
import matplotlib.pyplot as plt
for column in df.columns:
     if df[column].dtype != 'object':  #- OR-->if df[column].dtype == 'float':
        plt.figure()
        df.boxplot(column=column,vert=False)
        plt.show()


# In[21]:


#Identifying the type of relationship b/w features
#Scatter plot(multi variate analysis)
plt.scatter(df["Energy Usage"],df["CO2 Emissions"])
plt.show()



plt.figure()
plt.scatter(df["Ease of Business"],df["Business Tax Rate"])
plt.title("Ease of Business Vs Business Tax Rate")
plt.show()

plt.figure()
plt.scatter(df["Tourism Inbound"],df["GDP"])
plt.title("Tourism Vs GDP")
plt.show()


# In[22]:


#Population Distribution by Age Group
population_groups = ['Population 0-14', 'Population 15-64', 'Population 65+']
population_values = df[population_groups].sum()

plt.figure(figsize=(8, 8))
plt.pie(population_values, labels=population_groups, autopct='%1.1f%%', startangle=50)
plt.title('Population Distribution by Age Group')
plt.show()


# In[7]:


#Replacing outliers using capping method
def iqr_capping (df, cols, factor):
    for col in cols:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        
        IQR = Q3-Q1
        Upper_whisker = Q3 + (factor * IQR)
        Lower_whisker = Q1 - (factor * IQR)
        
        df[col] = np.where(df[col]<Lower_whisker,Lower_whisker,
                 np.where(df[col]>Upper_whisker,Upper_whisker,df[col]))
    return df


# In[ ]:


# Apply IQR capping 
cols_to_cap = ['Business Tax Rate', 'Health Exp % GDP', 'Days to Start Business', 'Ease of Business', 
               'Energy Usage','Population 65+','Population 15-64',
               'Mobile Phone Usage','Life Expectancy Male','Life Expectancy Female','Lending Interest','Hours to do Tax'] 

df = iqr_capping(df, cols_to_cap, factor=1.5)


# In[8]:


#filling null values in final_df if there are any
df.fillna(df.mean(),inplace=True)


# In[25]:


SS=StandardScaler()
for column in df.columns:
    if df[column].dtype!='object':
        df[column] = SS.fit_transform(df[[column]])
df 


# In[9]:


#Adding new features
df['GDP per Capita'] = df['GDP'] / df['Population Total']
df['Health Exp % GDP'] =df['Health Exp/Capita'] / df['GDP']
df['Tourism Ratio'] = df['Tourism Inbound'] / (df['Tourism Outbound'] + 1)


# In[10]:


df.shape


# In[11]:


df.info()


# In[ ]:




