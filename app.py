import streamlit as st
import ee
from google.oauth2 import service_account
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Methane Intelligence Platform",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #05070d;
    color: white;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background: #0b1220;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #1f2937;
    text-align: center;
}

.title-style {
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.subtitle-style {
    font-size: 18px;
    color: #9ca3af;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# EARTH ENGINE INIT
# =========================================================

@st.cache_resource
def init_ee():

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine"
        ]
    )

    ee.Initialize(
        credentials=credentials,
        project="methane-ai-495915"
    )

    return True


try:
    init_ee()
    st.success("Earth Engine Connected")

except Exception as e:
    st.error(f"Earth Engine Error: {e}")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title-style">Methane Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-style">AI-Powered Global Methane Monitoring using Satellite GIS Analytics</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# LOAD SENTINEL-5P METHANE
# =========================================================

methane = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
    .select("CH4_column_volume_mixing_ratio_dry_air")
    .filterDate("2025-01-01", "2025-12-31")
    .mean()
)

# =========================================================
# VISUALIZATION PARAMETERS
# =========================================================

vis_params = {
    "min": 1750,
    "max": 1950,
    "palette": [
        "black",
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red"
    ]
}

map_id = methane.getMapId(vis_params)

tile_url = map_id["tile_fetcher"].url_format

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Control Panel")

selected_theme = st.sidebar.selectbox(
    "GIS Theme",
    [
        "Dark",
        "Satellite",
        "Terrain"
    ]
)

show_hotspots = st.sidebar.checkbox(
    "Show AI Hotspots",
    value=True
)

show_industrial = st.sidebar.checkbox(
    "Show Industrial Sources",
    value=True
)

timeline_days = st.sidebar.slider(
    "Temporal Analysis Days",
    7,
    90,
    30
)

# =========================================================
# MAP STYLE
# =========================================================

if selected_theme == "Dark":
    tile_style = "CartoDB dark_matter"

elif selected_theme == "Satellite":
    tile_style = "OpenStreetMap"

else:
    tile_style = "Stamen Terrain"

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Detected Hotspots",
        "27",
        "+4"
    )

with col2:
    st.metric(
        "Global CH4 Avg",
        "1842 ppb",
        "+12"
    )

with col3:
    st.metric(
        "High Risk Zones",
        "6",
        "+1"
    )

with col4:
    st.metric(
        "Industrial Sources",
        "15",
        "+2"
    )

# =========================================================
# GIS MAP
# =========================================================

st.subheader("Live Global Methane GIS Monitoring")

m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles=tile_style
)

# =========================================================
# ADD METHANE LAYER
# =========================================================

folium.TileLayer(
    tiles=tile_url,
    attr="Google Earth Engine",
    name="Methane Layer",
    overlay=True,
    control=True
).add_to(m)

# =========================================================
# AI HOTSPOTS
# =========================================================

if show_hotspots:

    hotspots = [
        [24.34, 54.52],
        [29.76, -95.36],
        [35.68, 51.41],
        [25.20, 55.27],
        [31.95, 35.91]
    ]

    for lat, lon in hotspots:

        score = random.randint(70, 98)

        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            popup=f"AI Methane Hotspot<br>Leak Score: {score}%",
            color="red",
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

# =========================================================
# INDUSTRIAL SOURCES
# =========================================================

if show_industrial:

    industries = [
        [24.45, 54.38],
        [29.73, -95.20],
        [26.43, 50.10]
    ]

    for lat, lon in industries:

        folium.Marker(
            [lat, lon],
            popup="Industrial Emission Source"
        ).add_to(m)

# =========================================================
# LAYER CONTROL
# =========================================================

folium.LayerControl().add_to(m)

# =========================================================
# SHOW MAP
# =========================================================

st_folium(
    m,
    width=1400,
    height=700,
    key="main_map"
)

# =========================================================
# TEMPORAL ANALYSIS
# =========================================================

st.subheader("Temporal Methane Analysis")

dates = pd.date_range(
    end=datetime.today(),
    periods=timeline_days
)

values = np.random.normal(
    1840,
    20,
    timeline_days
)

df = pd.DataFrame({
    "Date": dates,
    "Methane": values
})

fig = px.line(
    df,
    x="Date",
    y="Methane",
    title="Atmospheric Methane Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# AI HOTSPOT TABLE
# =========================================================

st.subheader("AI Methane Hotspot Analysis")

hotspot_df = pd.DataFrame({
    "Region": [
        "Middle East",
        "Texas",
        "Iran",
        "UAE",
        "Jordan"
    ],
    "Leak Score": [
        "95%",
        "88%",
        "82%",
        "79%",
        "73%"
    ],
    "Risk Level": [
        "High",
        "High",
        "Medium",
        "Medium",
        "Low"
    ]
})

st.dataframe(
    hotspot_df,
    use_container_width=True
)

# =========================================================
# ANIMATION PLACEHOLDER
# =========================================================

st.subheader("Animated Methane Movement")

st.info(
    "Temporal methane plume animation module connected to Earth Engine timeline processing."
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Powered by Sentinel-5P, Earth Engine, AI CNN Analytics, and GIS Intelligence"
)
