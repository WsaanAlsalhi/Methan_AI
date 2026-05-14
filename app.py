import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
import folium
from streamlit_folium import st_folium
import ee
import requests
from PIL import Image
from io import BytesIO

from model import UNetPro


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Methane AI Monitoring System",
    layout="wide"
)

st.title("Methane AI GIS Monitoring Platform")


st.info(
    "Live integration with Google Earth Engine + Sentinel-5P methane observations."
)


# =====================================================
# EARTH ENGINE INIT (FIXED PROJECT ID)
# =====================================================
@st.cache_resource
def init_earth_engine():

    ee.Initialize(
        project="methane-ai-495915"
    )

    return "Connected"


connection_status = init_earth_engine()


# =====================================================
# MODEL LOAD
# =====================================================
@st.cache_resource
def load_model():

    model = UNetPro()

    model.load_state_dict(
        torch.load("best_model.pth", map_location="cpu")
    )

    model.eval()

    return model


model = load_model()


# =====================================================
# FETCH SENTINEL-5P DATA
# =====================================================
@st.cache_data
def fetch_data(country, start_date, end_date):

    regions = {
        "Oman": [52, 16, 60, 27],
        "Qatar": [50, 24, 52, 27],
        "Saudi Arabia": [34, 16, 56, 33],
        "UAE": [51, 22, 57, 27],
        "Global": [-180, -60, 180, 80]
    }

    bbox = regions[country]

    region = ee.Geometry.Rectangle(bbox)

    image = (
        ee.ImageCollection(
            "COPERNICUS/S5P/OFFL/L3_CH4"
        )
        .select("CH4_column_volume_mixing_ratio_dry_air")
        .filterDate(start_date, end_date)
        .mean()
    )

    url = image.getThumbURL({
        "min": 1750,
        "max": 1950,
        "region": region,
        "dimensions": 512,
        "palette": [
            "black",
            "blue",
            "cyan",
            "green",
            "yellow",
            "red"
        ]
    })

    response = requests.get(url)

    img = Image.open(BytesIO(response.content)).convert("L")

    img = np.array(img).astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img, bbox


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Control Panel")

st.sidebar.write("Earth Engine:", connection_status)

country = st.sidebar.selectbox(
    "Region",
    ["Oman", "Qatar", "Saudi Arabia", "UAE", "Global"]
)

start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

run = st.sidebar.button("Run Analysis")


# =====================================================
# MAIN
# =====================================================
if run:

    if connection_status != "Connected":
        st.error("Earth Engine not connected")
        st.stop()

    img, bbox = fetch_data(
        country,
        str(start_date),
        str(end_date)
    )

    # =========================
    # PREPROCESS
    # =========================
    input_img = resize(img, (256, 256))

    tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).float()

    # =========================
    # CNN PREDICTION
    # =========================
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)

    mask = pred.squeeze().numpy()
    binary = mask > 0.4

    confidence = float(np.mean(mask))

    # =========================
    # OVERVIEW
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Satellite", "Sentinel-5P")
    col2.metric("Model", "CNN (U-Net)")
    col3.metric("Confidence", round(confidence, 2))

    st.divider()

    # =========================
    # IMAGE
    # =========================
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Detection Map")

        fig, ax = plt.subplots(figsize=(4,4))

        ax.imshow(input_img, cmap="viridis")

        ax.imshow(binary, cmap="Reds", alpha=0.5)

        ax.contour(binary, colors="yellow", linewidths=1)

        ax.axis("off")

        st.pyplot(fig)

        if confidence > 0.2:
            st.error("Methane anomaly detected")
        else:
            st.success("No anomaly detected")

    # =========================
    # GIS MAP
    # =========================
    with col2:

        st.subheader("GIS View")

        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="red",
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=6,
            color="red",
            fill=True
        ).add_to(m)

        st_folium(m, width=400, height=400)

else:
    st.info("Select region and run analysis")