import streamlit as st
import ee
from google.oauth2 import service_account
import folium
from streamlit_folium import st_folium


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Methane AI GIS System",
    layout="wide"
)


# -----------------------------
# EARTH ENGINE INIT (FIXED)
# -----------------------------
@st.cache_resource
def init_earth_engine():

    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        ee.Initialize(
            credentials=credentials,
            project="methane-ai-495915"
        )

        return "Earth Engine Connected"

    except Exception as e:
        return f"Connection Failed: {str(e)}"


status = init_earth_engine()
st.sidebar.success(status)


# -----------------------------
# SIMPLE GLOBAL MAP (GIS BASE)
# -----------------------------
st.title("Methane Detection GIS Dashboard")


m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

# Example methane hotspot markers (dummy demo)
hotspots = [
    [24.7136, 46.6753],   # Riyadh
    [23.5859, 58.4059],   # Oman
    [25.276987, 55.296249] # UAE
]

for i, point in enumerate(hotspots):
    folium.CircleMarker(
        location=point,
        radius=8,
        color="red",
        fill=True,
        fill_opacity=0.6,
        popup=f"Methane Hotspot {i+1}"
    ).add_to(m)


# IMPORTANT: avoid duplicate keys error
map_data = st_folium(
    m,
    width=1000,
    height=500,
    key="main_map"
)


# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.title("System Info")
st.sidebar.write("AI Methane Detection Platform")
st.sidebar.write("Powered by CNN + Earth Engine")
st.sidebar.write("Status: Production Mode")
