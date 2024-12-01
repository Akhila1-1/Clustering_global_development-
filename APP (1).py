#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load Data
st.title("Global Development Clustering")
st.sidebar.header("Options")
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/e/e5/World_map_blank_without_borders.png');
            background-size: cover;
        }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\maheh\\Downloads\\World_development_mesurement (1).csv")
    # Impute missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Clean and scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df, X_scaled

df, X_scaled = load_data()

# Sidebar: Select clustering parameters
linkage_method = st.sidebar.selectbox("Select Linkage Method", ["single", "average", "complete", "ward"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

# Perform clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
labels = clustering.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, labels)
st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", ax=ax)
ax.set_title("Clusters Visualized with PCA")
st.pyplot(fig)

# Show data insights
st.subheader("Clustered Data")
df['Cluster'] = labels
st.dataframe(df.head())

# Additional Plot
st.subheader("Explore Relationships")
feature_x = st.sidebar.selectbox("Select X-axis Feature", df.select_dtypes(include=[np.number]).columns)
feature_y = st.sidebar.selectbox("Select Y-axis Feature", df.select_dtypes(include=[np.number]).columns)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='Cluster', palette="Set2", ax=ax)
st.pyplot(fig)

# Download clustered data
st.download_button("Download Clustered Data", data=df.to_csv(index=False), file_name="clustered_data.csv")


# In[ ]:




