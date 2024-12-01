#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# In[5]:


df=pd.read_csv("C:\\Users\\maheh\\Downloads\\World_development_mesurement (1).csv")
df


# In[6]:


#Imputation
for columns in df.columns:
    if df[columns].dtype == 'int':
        df[columns].fillna(df[columns].mean(),inplace=True)
    if df[columns].dtype == 'float':
        df[columns].fillna(df[columns].mean(),inplace=True)
    else:
        df[columns].fillna(df[columns].mode()[0],inplace=True)
df


# In[7]:


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


# In[8]:


#dropping feature No.of records feature as it'st.deviation is zero
#df.drop(columns=["Number of Records"],inplace=True)
#df.drop(columns=["Country"],inplace=True)


# In[9]:


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


# In[10]:


# Apply IQR capping 
cols_to_cap = ['Business Tax Rate', 'Health Exp % GDP', 'Days to Start Business', 'Ease of Business', 
               'Energy Usage','Population 65+','Population 15-64',
               'Mobile Phone Usage','Life Expectancy Male','Life Expectancy Female','Lending Interest','Hours to do Tax'] 

df = iqr_capping(df, cols_to_cap, factor=1.5)


# In[11]:


#filling null values in final_df if there are any
df.fillna(df.mean(),inplace=True)


# In[12]:


#Adding new features
df['GDP per Capita'] = df['GDP'] / df['Population Total']
df['Health Exp % GDP'] =df['Health Exp/Capita'] / df['GDP']
df['Tourism Ratio'] = df['Tourism Inbound'] / (df['Tourism Outbound'] + 1)


# In[13]:


scaler = StandardScaler()
X1 = scaler.fit_transform(df.select_dtypes(include=[np.number]))


# In[26]:


#hierarchical Clustering with different linkage methods 
linkage_methods = ['single', 'average', 'complete', 'ward'] 
for method in linkage_methods: 
    clustering = AgglomerativeClustering(n_clusters=3, linkage=method) 
    labels = clustering.fit_predict(X1) 
    score = silhouette_score(X1, labels) 
    print(f'Silhouette Score for {method} linkage: {score:.2f}')
    


# In[37]:


#Agglomerative clustering Linkage="ward"
cluster=AgglomerativeClustering(n_clusters=3,linkage='single')
agg_single=cluster.fit_predict(X1)
df.head()

score=silhouette_score(X1,agg_single)
print("silhoutte Score for ward linkage is:",score.round(2))


# In[40]:


# Step 4: Visualize the Clusters

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X1)

plt.figure(figsize=(10, 5))
sns.scatterplot(X_pca[:, 0], X_pca[:, 1], hue=agg_single, palette='Set2', alpha=0.6)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='purple', label='Centroids')
plt.title('Hierarchial Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




