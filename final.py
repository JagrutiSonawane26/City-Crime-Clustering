# Generated from: final.ipynb
# Converted at: 2026-03-24T07:57:10.990Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv("US_Crime_Data.csv")

df.head()

df.shape

df.columns

df.info()

df.describe()

df.isnull().sum()

df = df.drop(columns=['URL'])

df.info()

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M', errors='coerce')

df = df.dropna(subset=['Date'])

df.info()

df[['City', 'State', 'Title']] = df[['City', 'State', 'Title']].fillna('Unknown')

df = df.drop_duplicates()

df.info()

df['Hour'] = df['Date'].dt.hour
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

df['City_Freq'] = df['City'].map(df['City'].value_counts())
df['State_Freq'] = df['State'].map(df['State'].value_counts())

tfidf = TfidfVectorizer(stop_words='english', max_features=100)
text_matrix = tfidf.fit_transform(df['Title'])

svd = TruncatedSVD(n_components=3, random_state=42)
text_features = svd.fit_transform(text_matrix)

text_df = pd.DataFrame(text_features, columns=['Text_1', 'Text_2', 'Text_3'])

numeric_cols = ['Hour', 'DayOfWeek', 'Is_Weekend', 'City_Freq', 'State_Freq']
X_numeric = df[numeric_cols].reset_index(drop=True)
X_combined = pd.concat([X_numeric, text_df], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

k_sil = silhouette_score(X_scaled, df['KMeans_Cluster'])

dbscan = DBSCAN(eps=2.0, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

db_labels = df['DBSCAN_Cluster']

if len(set(db_labels)) > 1:
    db_sil = silhouette_score(X_scaled, db_labels)
else:
    db_sil = 0

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(16, 12))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['KMeans_Cluster'],
    palette='viridis',
    s=50
)
plt.title(f"K-Means Clustering (Silhouette: {k_sil:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.subplot(1, 2, 2)
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['DBSCAN_Cluster'],
    palette='tab10',
    s=50
)
plt.title(f"DBSCAN Clustering (Silhouette: {db_sil:.2f})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(X_numeric.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

inertia = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(range(2, 9), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

print("\n--- FINAL PROJECT REPORT ---")
print(f"K-Means Silhouette Score: {k_sil:.4f}")
print(f"DBSCAN Silhouette Score: {db_sil:.4f}")
print(f"KMeans Clusters: {set(df['KMeans_Cluster'])}")
print(f"DBSCAN Clusters: {set(map(int, df['DBSCAN_Cluster']))}")

print("\nCluster-wise Analysis (KMeans):")
print(df.groupby('KMeans_Cluster').mean(numeric_only=True))

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="crime_mapper")
df['Coordinates'] = df['City'] + ", " + df['State']
df['Location'] = df['Coordinates'].apply(lambda x: geolocator.geocode(x))
df['Latitude'] = df['Location'].apply(lambda loc: loc.latitude if loc else np.nan)
df['Longitude'] = df['Location'].apply(lambda loc: loc.longitude if loc else np.nan)