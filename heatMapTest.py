import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

df = pd.read_csv("nhgis0001_ds258_2020_block.csv")

selected_columns = df[['INTPTLAT', 'INTPTLON', df.columns[-1]]]  
selected_columns = selected_columns.dropna()

selected_columns[df.columns[-1]] = np.log1p(selected_columns[df.columns[-1]])

first_point = selected_columns.iloc[0]
location = [first_point['INTPTLAT'], first_point['INTPTLON']]

m = folium.Map(location=location, zoom_start=5)
HeatMap(data=selected_columns[['INTPTLAT', 'INTPTLON', df.columns[-1]]].values.tolist()).add_to(m)
m.save("heatmap.html")

top_20 = selected_columns.nlargest(20, df.columns[-1])
print(top_20)
