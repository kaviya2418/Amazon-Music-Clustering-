🎵 Amazon Music Clustering Project
📌 Project Overview

With millions of songs available on music streaming platforms like Amazon Music, manually categorizing tracks into genres is inefficient.
This project applies unsupervised machine learning techniques to automatically group songs based on their audio characteristics, without using predefined genre labels.

The system clusters songs using features such as danceability, energy, tempo, loudness, and mood, helping uncover hidden patterns that represent musical styles or listening moods.

🎯 Objectives

Automatically group similar songs using clustering algorithms

Identify meaningful music clusters based on audio features

Detect musically unique songs (outliers)

Visualize high-dimensional music data using PCA

Build an interactive Streamlit dashboard for exploration

🧠 Domain

Music Analytics / Unsupervised Machine Learning

🛠 Skills & Tools Used

Data Exploration & Cleaning

Feature Selection & Normalization

K-Means, DBSCAN, Hierarchical Clustering

Elbow Method & Silhouette Score

Davies–Bouldin Index

PCA (Principal Component Analysis)

Data Visualization (Matplotlib, Plotly)

Python (Pandas, NumPy, scikit-learn)

Streamlit (Dashboard Development)

📂 Dataset Description

File: single_genre_artists.csv

Type: Audio feature dataset from Amazon Music

Key Features:

danceability

energy

loudness

speechiness

acousticness

instrumentalness

liveness

valence

tempo

duration_ms

Reference Columns: track name, artist name, IDs (not used for clustering)

These features describe rhythm, mood, intensity, and instrumentation of songs.

🔍 Project Workflow
1️⃣ Data Exploration & Preprocessing

Loaded dataset and examined structure

Checked missing values and duplicates

Removed non-numeric and identifier columns

Scaled features using StandardScaler

2️⃣ Feature Selection

Selected audio features that best represent how a song sounds, including rhythm, energy, and mood.

3️⃣ Clustering Techniques

K-Means Clustering

Used Elbow Method and Silhouette Score to determine optimal clusters

DBSCAN

Identified dense clusters and detected noise (outliers)

Hierarchical Clustering

Visualized cluster hierarchy using dendrograms

4️⃣ Evaluation Metrics

Silhouette Score

Davies–Bouldin Index

Cluster size distribution

Noise percentage (for DBSCAN)

5️⃣ Dimensionality Reduction

Applied PCA for 2D and 3D visualization of clusters

6️⃣ Visualization

PCA scatter plots

Cluster comparison bar charts

Heatmaps of feature intensity

Violin plots for feature distribution

7️⃣ Final Output

Cluster labels added to dataset

Exported final results to CSV

Built an interactive Streamlit dashboard

📊 Results & Insights

Successfully formed distinct clusters of songs based on audio similarity

Identified:

High-energy / dance tracks

Calm / acoustic tracks

Musically unique songs as outliers

DBSCAN effectively detected noise tracks

PCA helped visualize cluster separation clearly

📈 Business Use Cases

🎧 Personalized playlist generation

🔍 Improved song discovery & recommendations

🎤 Artist and producer competitive analysis

📢 Market segmentation for promotions

🖥 Streamlit Dashboard

An interactive dashboard was built using Streamlit to:

Visualize clusters in 3D PCA space

Explore feature distributions by cluster

Analyze outlier songs

Interact with musical features dynamically

Run the app:

streamlit run amazon.py
📁 Project Structure
Amazon-Music-Clustering/
│
├── single_genre_artists.csv
├── amazon_music_final_clusters.csv
├── amazon_music_clustering.ipynb
├── amazon.py
├── README.md
📦 Project Deliverables

✔ Jupyter Notebook with full analysis

✔ CSV file with cluster labels

✔ Streamlit interactive dashboard

✔ Final report / documentation

🚀 Future Enhancements

Add genre prediction using labeled datasets

Implement recommendation engine

Deploy Streamlit app on Streamlit Cloud

Integrate user listening history
