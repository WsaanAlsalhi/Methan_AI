import streamlit as st
import ee
from google.oauth2 import service_account
import folium
from streamlit_folium import st_folium


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Methane Monitoring System",
    layout="wide"
)


# =========================================
# EARTH ENGINE AUTH
# =========================================

@st.cache_resource
def init_ee():

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    ee.Initialize(
        credentials=credentials,
        project="methane-ai-495915"
    )

    return "Earth Engine Connected"


status = init_ee()

st.sidebar.success(status)


# =========================================
# TITLE
# =========================================

st.title("Global Methane Monitoring GIS Platform")

st.markdown(
    """
    Real-time atmospheric methane monitoring using Sentinel-5P satellite data,
    Earth Engine cloud processing, and AI-based anomaly analysis.
    """
)


# =========================================
# LOAD SENTINEL-5P METHANE
# =========================================

methane = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
    .select("CH4_column_volume_mixing_ratio_dry_air")
    .filterDate("2025-01-01", "2025-12-31")
    .mean()
)


# =========================================
# VISUAL PARAMETERS
# =========================================

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


# =========================================
# GET MAP TILES
# =========================================

map_id = methane.getMapId(vis_params)

tile_url = map_id["tile_fetcher"].url_format


# =========================================
# CREATE GIS MAP
# =========================================

m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles="CartoDB dark_matter"
)


# =========================================
# ADD METHANE LAYER
# =========================================

folium.TileLayer(
    tiles=tile_url,
    attr="Google Earth Engine",
    name="Methane Concentration",
    overlay=True,
    control=True
).add_to(m)


# =========================================
# LAYER CONTROL
# =========================================

folium.LayerControl().add_to(m)


# =========================================
# SHOW MAP
# =========================================

st_folium(
    m,
    width=1400,
    height=700,
    key="methane_map"
)
