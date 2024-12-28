import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Function to set background image
def set_background(image_file):
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_file}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Load and encode background image
def load_image(image_path):
    with open(image_path, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    return encoded_string

# Apply background image
background_image = load_image("C:\\Users\\maheh\\Downloads\\Untitled design (1).png")  # Replace with your image filename
set_background(background_image)


# Title and description
st.title("Global Development Clustering App 🌍")
st.write("""
This app performs hierarchical clustering on global development data to group countries into development categories such as 
Low, Moderate, and High Development based on socio-economic indicators.
""")


# Load the dataset
@st.cache_data
def load_data():
    # Load the preprocessed dataset
    df = pd.read_csv("C:\\Users\\maheh\\Downloads\\World_development_mesurement (1).csv")
    
    # Cleaning data
    columns_to_clean = ['GDP', 'Health Exp/Capita', 'Business Tax Rate', 'Tourism Inbound', 'Tourism Outbound']
    for column in columns_to_clean:
        df[column] = df[column].astype(str).str.replace(r'[$%]', '', regex=True).str.replace(r'[^\d.]', '', regex=True).replace('', '0').astype(float)

    # Impute missing values
    df.fillna(df.mean(), inplace=True)

    # Add derived features
    df['GDP per Capita'] = df['GDP'] / df['Population Total'].replace(0, np.nan)
    df['Health Exp % GDP'] = df['Health Exp/Capita'] / df['GDP'].replace(0, np.nan)
    df['Tourism Ratio'] = df['Tourism Inbound'] / (df['Tourism Outbound'] + 1).replace(0, np.nan)

    # Replace infinities and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # Drop unnecessary columns
    return df


# Preprocess and cluster the data
df = load_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Perform Hierarchical Clustering
st.write("### Performing Clustering")
n_clusters = 3  # Fixed number of clusters
linkage_method = 'single'
cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
labels = cluster.fit_predict(X_scaled)
df['Cluster'] = labels

# Map clusters to development categories
cluster_map = {0: "Low Development", 1: "Moderate Development", 2: "High Development"}
df['Development Category'] = df['Cluster'].map(cluster_map)

# Display the clustered data
st.write("### Clustered Data with Development Categories")
st.dataframe(df[['Country', 'Development Category']] if 'Country' in df.columns else df[['Development Category']])

# Show cluster summary
st.write("### Cluster Summary")
cluster_summary = df.groupby('Development Category').mean()
st.dataframe(cluster_summary)

# Visualize clusters with PCA
st.write("### Cluster Visualization")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', alpha=0.6, ax=ax)
plt.title("Clusters Visualized in PCA Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
st.pyplot(fig)

# Country-specific cluster details
st.write("### Country Development Details")
if 'Country' in df.columns:
    country_name = st.text_input("Enter a country name:")
    if country_name:
        if country_name in df['Country'].values:
            country_info = df[df['Country'] == country_name]
            st.write(f"**Country:** {country_name}")
            st.write(f"**Development Category:** {country_info['Development Category'].values[0]}")
        else:
            st.write("Country not found in the dataset.")


st.text("Made with 💜 by our team")
