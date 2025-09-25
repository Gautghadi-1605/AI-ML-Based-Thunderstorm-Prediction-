import cdsapi
import os

# Set working directory to the folder where your script is
os.chdir(r"C:\thunder")

# Initialize CDS API client
client = cdsapi.Client()

# Define dataset and request
dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure"
    ],
    "year": "2024",
    "month": "08",
    "day": "01",
    "time": ["00:00","06:00","12:00","18:00"],  # 4 timesteps
    "area": [30, 70, 20, 80],  # north, west, south, east (small region)
    "format": "grib"
}

# Download the dataset to the same folder as script
client.retrieve(dataset, request).download("era5_sample_demo.grib")

print("Download complete! File saved as C:\\thunder\\era5_sample_demo.grib")
