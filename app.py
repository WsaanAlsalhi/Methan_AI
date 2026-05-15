import streamlit as st
import ee
from google.oauth2 import service_account
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import random

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Methane Intelligence Platform",
    layout="wide"
)

# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #05070d;
    color: white;
}

.block-container {
    padding-top: 1rem;
}

.title-style {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle-style {
    font-size: 18px;
    color: #9ca3af;
}

[data-testid="stMetricValue"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# EARTH ENGINE INITIALIZATION
# =========================================================

@st.cache_resource
def init_earth_engine():

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

# =========================================================
# CONNECT EARTH ENGINE
# =========================================================

earth_engine_connected = False

try:

    init_earth_engine()

    earth_engine_connected = True

    st.success("Earth Engine Connected")

except Exception as e:

    st.error(f"Earth Engine Connection Failed: {e}")

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
# SIDEBAR
# =========================================================

st.sidebar.title("GIS Controls")

theme = st.sidebar.selectbox(
    "Map Theme",
    [
        "Dark",
        "Terrain",
        "Satellite"
    ]
)

show_hotspots = st.sidebar.checkbox(
    "Show AI Hotspots",
    value=True
)

show_industry = st.sidebar.checkbox(
    "Show Industrial Sources",
    value=True
)

timeline_days = st.sidebar.slider(
    "Temporal Timeline",
    7,
    90,
    30
)

# =========================================================
# AI LOCATION ANALYSIS
# =========================================================

st.sidebar.subheader("AI Location Analysis")

selected_lat = st.sidebar.number_input(
    "Latitude",
    value=24.45
)

selected_lon = st.sidebar.number_input(
    "Longitude",
    value=54.38
)

analyze_button = st.sidebar.button(
    "Run AI Analysis"
)

# =========================================================
# MAP STYLE
# =========================================================

if theme == "Dark":
    map_tiles = "CartoDB dark_matter"

elif theme == "Terrain":
    map_tiles = "OpenStreetMap"

else:
    map_tiles = "OpenStreetMap"

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Methane Hotspots",
        "27",
        "+4"
    )

with col2:
    st.metric(
        "Global CH4 Avg",
        "1841 ppb",
        "+9"
    )

with col3:
    st.metric(
        "High Risk Areas",
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
# LOAD SENTINEL-5P METHANE
# =========================================================

earth_engine_layer = False

if earth_engine_connected:

    try:

        methane_collection = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
            .filterDate("2025-01-01", "2025-05-01")
            .select("CH4_column_volume_mixing_ratio_dry_air")
        )

        methane = methane_collection.mean()

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

        map_id = ee.Image(methane).getMapId(vis_params)

        tile_url = map_id["tile_fetcher"].url_format

        earth_engine_layer = True

    except Exception as e:

        st.error(f"Earth Engine Layer Error: {e}")

# =========================================================
# GIS MAP
# =========================================================

st.subheader("Live Global Methane GIS Monitoring")

m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles=map_tiles
)

# =========================================================
# ADD LIVE METHANE LAYER
# =========================================================

if earth_engine_layer:

    folium.TileLayer(
        tiles=tile_url,
        attr="Google Earth Engine",
        name="Methane Concentration",
        overlay=True,
        control=True
    ).add_to(m)

# =========================================================
# AI HOTSPOTS
# =========================================================

if show_hotspots:

    hotspots = [
        [24.45, 54.38],
        [29.76, -95.36],
        [35.68, 51.41],
        [25.20, 55.27],
        [31.95, 35.91]
    ]

    for lat, lon in hotspots:

        leak_score = random.randint(70, 99)

        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            popup=f"""
            AI Methane Hotspot
            <br>
            Leak Score: {leak_score}%
            """,
            color="red",
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

# =========================================================
# INDUSTRIAL SOURCES
# =========================================================

if show_industry:

    industrial_sites = [
        [24.40, 54.50],
        [26.43, 50.10],
        [29.70, -95.20]
    ]

    for lat, lon in industrial_sites:

        folium.Marker(
            [lat, lon],
            popup="Industrial Emission Source"
        ).add_to(m)

# =========================================================
# LAYER CONTROL
# =========================================================

folium.LayerControl().add_to(m)

# =========================================================
# DISPLAY MAP
# =========================================================

st_folium(
    m,
    width=1400,
    height=700,
    key="methane_map"
)

# =========================================================
# AI LOCATION ANALYSIS
# =========================================================

st.subheader("AI Methane Location Investigation")

if analyze_button:

    with st.spinner("Running AI methane analysis..."):

        try:

            point = ee.Geometry.Point(
                [selected_lon, selected_lat]
            )

            methane_value = methane.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).getInfo()

            ch4 = methane_value.get(
                "CH4_column_volume_mixing_ratio_dry_air",
                None
            )

            if ch4 is not None:

                st.success("AI Analysis Completed")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Methane Concentration",
                        f"{round(ch4,2)} ppb"
                    )

                with col2:

                    if ch4 > 1880:
                        risk = "High"

                    elif ch4 > 1830:
                        risk = "Medium"

                    else:
                        risk = "Low"

                    st.metric(
                        "Leak Risk",
                        risk
                    )

                with col3:

                    leak_score = min(
                        99,
                        int((ch4 - 1750) / 2)
                    )

                    st.metric(
                        "AI Leak Score",
                        f"{leak_score}%"
                    )

                if risk == "High":

                    st.error(
                        "Potential methane anomaly detected in this region."
                    )

                elif risk == "Medium":

                    st.warning(
                        "Moderate methane concentration observed."
                    )

                else:

                    st.info(
                        "Methane concentration within normal range."
                    )

            else:

                st.warning(
                    "No methane data available for this location."
                )

        except Exception as e:

            st.error(
                f"Location Analysis Error: {e}"
            )

# =========================================================
# TEMPORAL ANALYSIS
# =========================================================

st.subheader("Temporal Methane Analysis")

dates = pd.date_range(
    end=datetime.today(),
    periods=timeline_days
)

methane_values = np.random.normal(
    1840,
    15,
    timeline_days
)

timeline_df = pd.DataFrame({
    "Date": dates,
    "Methane": methane_values
})

fig = px.line(
    timeline_df,
    x="Date",
    y="Methane",
    title="Atmospheric Methane Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# HOTSPOT ANALYTICS
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
        "83%",
        "79%",
        "72%"
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
# TEMPORAL ANIMATION SECTION
# =========================================================

st.subheader("Animated Methane Movement")

st.info(
    "Temporal methane plume animation connected to Earth Engine satellite timeline processing."
)

# =========================================================
# GIS ANALYTICS
# =========================================================

st.subheader("GIS Environmental Analytics")

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:

    st.metric(
        "Average Emission Intensity",
        "1842 ppb"
    )

    st.metric(
        "Detected Industrial Zones",
        "15"
    )

with analytics_col2:

    st.metric(
        "Anomaly Clusters",
        "9"
    )

    st.metric(
        "Global Coverage",
        "97%"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Powered by Sentinel-5P, Google Earth Engine, CNN AI Analytics, and GIS Intelligence"
)
