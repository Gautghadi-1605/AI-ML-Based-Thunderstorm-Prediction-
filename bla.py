import pandas as pd

# File paths
files = {
    "pressure": r"C:\thunder\archive (8)\pressure.csv",
    "humidity": r"C:\thunder\archive (8)\humidity.csv",
    "temperature": r"C:\thunder\archive (8)\temperature.csv",
    "weather": r"C:\thunder\archive (8)\weather_description.csv",
    "wind_dir": r"C:\thunder\archive (8)\wind_direction.csv",
    "wind_speed": r"C:\thunder\archive (8)\wind_speed.csv"
}

# Read and reshape each CSV
dfs = {}
for key, path in files.items():
    df = pd.read_csv(path)
    df_long = df.melt(id_vars="datetime", var_name="city", value_name=key)
    dfs[key] = df_long

# Merge all dataframes on datetime and city
from functools import reduce

merged_df = reduce(lambda left, right: pd.merge(left, right, on=["datetime", "city"]), dfs.values())

# Preview
print(merged_df.head())

# Save to a new CSV
merged_df.to_csv(r"C:\thunder\merged_weather_data.csv", index=False)

