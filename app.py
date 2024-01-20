import base64
import io
import random
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from shapely.geometry import Point

# Loading the model and setting up metadata
model = PARSynthesizer.load(filepath="9_24_2023.pkl")
meta_input = pd.read_csv("final_input.csv")
meta_input["Date"] = pd.to_datetime(meta_input.Date)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(meta_input)
metadata.update_column(column_name="Well_UUID", sdtype="id")
metadata.set_sequence_key("Well_UUID")
metadata.set_sequence_index("Date")


@st.cache_data
def generate_data(num_wells, _gdf, _model):
    all_points = []
    for _, row in _gdf.iterrows():
        all_points.extend(random_points_in_polygon(num_wells, row["geometry"]))

    while len(all_points) > num_wells:
        all_points.pop()

    synthetic_data = _model.sample(num_wells)
    point_data = pd.DataFrame(
        {
            "Well_UUID": synthetic_data["Well_UUID"].unique().tolist(),
            "geometry": all_points,
        }
    )
    geosynth_data = pd.merge(synthetic_data, point_data, on="Well_UUID", how="inner")
    geosynth_data = gpd.GeoDataFrame(geosynth_data, geometry="geometry")
    geosynth_data = geosynth_data.sort_values(by=["Well_UUID", "Date"])
    return geosynth_data

def create_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'

def random_points_in_polygon(number, polygon):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    while len(points) < number:
        random_point = Point(
            [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        )
        if random_point.within(polygon):
            points.append(random_point)
    return points


# Streamlit app
st.title("🔮 Synthetic Well Data Generator using SDV")
st.write(
    """
Generate synthetic well data using pre-trained models from the Synthetic Data Vault (SDV) and your own shapefile. 
Upload your shapefile, adjust the parameters, and generate well data!
"""
)

shapefile = st.file_uploader(
    "Upload your Shapefile (should include .shp, .shx, .dbf, and any other necessary files as a zip archive)",
    type=["zip"],
)

gdf = None

if shapefile:
    with st.spinner("Loading shapefile..."):
        with io.BytesIO(shapefile.getvalue()) as unzip:
            gdf = gpd.read_file(unzip)
        st.write("Shapefile loaded successfully!")

    num_wells = st.slider("Select number of synthetic wells:", 1, 100, value=100)

    if "data_generated" not in st.session_state:
        st.session_state.data_generated = False

    if st.button("Generate Well Data") or st.session_state.data_generated:
        st.session_state.data_generated = True
        geosynth_data = generate_data(num_wells, gdf, model)
        quality_report = evaluate_quality(
            real_data=meta_input,
            synthetic_data=geosynth_data[
                ["Date", "Well_UUID", "GW_measurement_smoothed"]
            ],
            metadata=metadata,
        )
        quality_score = f"Quality Score: {round(quality_report.get_score()*100)}"
        st.write(quality_score)
        st.subheader("Generated Synthetic Well Data:")

        st.write(geosynth_data.head(10))
        st.markdown(create_download_link(geosynth_data), unsafe_allow_html=True)

        selected_well_uuid = st.selectbox(
            "Select a Well UUID to view its time series:",
            geosynth_data["Well_UUID"].unique().tolist(),
        )

        timeseries_data = geosynth_data[
            geosynth_data["Well_UUID"] == selected_well_uuid
        ]

        # Plotting logic
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            pd.to_datetime(timeseries_data["Date"]),
            timeseries_data["GW_measurement_smoothed"],
        )
        ax.set_title(f"Timeseries data for Well: {selected_well_uuid}")
        ax.set_xlabel("Date")
        ax.set_ylabel("GW Measurement Smoothed")
        st.pyplot(fig)
