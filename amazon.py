# =========================================================
#  Amazon Music Clustering Dashboard (Streamlit App)
# =========================================================

# ======================
# 1. IMPORT LIBRARIES
# ======================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ======================
# 2. PAGE CONFIGURATION
# ======================

st.set_page_config(
    page_title="Amazon Music Clustering Universe",
    page_icon="🎵",
    layout="wide"
)

# ======================
# 3. CUSTOM UI STYLING
# ======================

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle, #ffffff 0%, #f0f2f6 100%);
    }
    .main-card {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        border-radius: 22px;
        padding: 35px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px;
    }
    .kpi-box {
        background: white;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    .kpi-val {
        font-size: 30px;
        font-weight: 900;
        color: #4f46e5;
    }
    .kpi-lab {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 4. LOAD DATA
# ======================

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\KAVIYA V\Downloads\amazon_music_final_clusters.xls")

df = load_data()

# ======================
# 5. FEATURE LIST (MATCHES NOTEBOOK)
# ======================

features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

# ======================
# 6. CLUSTER COLUMN SELECTION
# ======================

# Priority: DBSCAN → KMeans → fallback
if 'db_cluster' in df.columns:
    df['Cluster'] = df['db_cluster']
elif 'kmeans_cluster' in df.columns:
    df['Cluster'] = df['kmeans_cluster']
else:
    st.error("No clustering column found in dataset.")
    st.stop()

# Noise and cluster counts
n_noise = (df['Cluster'] == -1).sum()
core_clusters = sorted([c for c in df['Cluster'].unique() if c != -1])
n_clusters = len(core_clusters)

# ======================
# 7. PCA COMPUTATION (IF NOT SAVED)
# ======================

if not {'PCA1', 'PCA2', 'PCA3'}.issubset(df.columns):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].fillna(0))

    pca = PCA(n_components=3, random_state=42)
    pca_vals = pca.fit_transform(X_scaled)

    df['PCA1'] = pca_vals[:, 0]
    df['PCA2'] = pca_vals[:, 1]
    df['PCA3'] = pca_vals[:, 2]

# ======================
# 8. HEADER SECTION
# ======================

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        width=110
    )

with col_title:
    st.title("🌌 Amazon Music Clustering Discovery")
    st.caption("Unsupervised Learning • DBSCAN • PCA Visualization")

# ======================
# 9. KPI DASHBOARD
# ======================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-lab">Total Tracks</div>
        <div class="kpi-val">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-lab">Audio Features</div>
        <div class="kpi-val">{len(features)}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="kpi-box">
        <div class="kpi-lab">Algorithm</div>
        <div class="kpi-val">DBSCAN</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-lab">Core Clusters</div>
        <div class="kpi-val">{n_clusters}</div>
    </div>
    """, unsafe_allow_html=True)

# ======================
# 10. CLUSTER LABELS
# ======================

def map_cluster(x):
    return "🌪️ Noise (Outliers)" if x == -1 else f"🎵 Cluster {int(x)}"

df['Cluster_Type'] = df['Cluster'].apply(map_cluster)

# ======================
# 11. 3D PCA VISUALIZATION
# ======================

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🌠 3D PCA Cluster Space")

# Sampling for performance
df_noise = df[df['Cluster'] == -1]
df_core = df[df['Cluster'] != -1].sample(
    min(5000, len(df[df['Cluster'] != -1])),
    random_state=42
)

df_viz = pd.concat([df_noise, df_core])

fig_3d = px.scatter_3d(
    df_viz,
    x='PCA1', y='PCA2', z='PCA3',
    color='Cluster_Type',
    opacity=0.8,
    template="plotly_white",
    height=750
)

fig_3d.update_layout(
    margin=dict(l=0, r=0, b=0, t=30),
    legend_title="Clusters"
)

st.plotly_chart(fig_3d, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 12. FEATURE COMPARISON
# ======================

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("📊 Feature Distribution by Cluster")

cluster_means = df.groupby('Cluster_Type')[features].mean().reset_index()
df_melt = cluster_means.melt(id_vars='Cluster_Type')

fig_bar = px.bar(
    df_melt,
    x='variable',
    y='value',
    color='Cluster_Type',
    barmode='group',
    template="plotly_white"
)

st.plotly_chart(fig_bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 13. FEATURE DEEP DIVE
# ======================

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("📈 Feature Distribution (Violin Plot)")

feature_selected = st.selectbox("Select audio feature:", features)

df_sample = df.groupby('Cluster', group_keys=False).apply(
    lambda x: x.sample(min(1500, len(x)), random_state=42)
)

fig_violin = px.violin(
    df_sample,
    x='Cluster_Type',
    y=feature_selected,
    color='Cluster_Type',
    box=True,
    points=False,
    template="plotly_white"
)

st.plotly_chart(fig_violin, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# 14. NOISE INSPECTION
# ======================

st.subheader(f"🌪️ Noise / Outlier Songs (Total: {n_noise:,})")

cols_to_show = [
    c for c in ['name_song', 'name_artists'] + features
    if c in df.columns
]

st.dataframe(
    df[df['Cluster'] == -1][cols_to_show].head(50),
    use_container_width=True
)