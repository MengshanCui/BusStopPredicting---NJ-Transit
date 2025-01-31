import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
tqdm.pandas()


# similar struchture to training.py

def color_mapping(actual, predicted):
    residual = -( actual - predicted ) 
    norm = Normalize(vmin=0, vmax=residual.max())
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(residual)  
    return colors

# similar R2 with all the data
stopsUrl = "Bus_stop\\stops.txt"
popuUrl = "nhgis0001_ds258_2020_block.csv"

def unifyDf(url):
    file_extension = os.path.splitext(url)[1]
    if file_extension in ['.txt', '.csv']:
        df = pd.read_csv(url)
    else:
        print("Unsupported file format.")
        return

    lat_col = [col for col in df.columns if 'lat' in col.lower()]
    lon_col = [col for col in df.columns if 'lon' in col.lower()]
    popu_col = [col for col in df.columns if 'u7h001' in col.lower()]

    if lat_col and lon_col:
        if popu_col:
            df = df[df[popu_col[0]] >= 1].copy()
            df[popu_col[0]] = np.log(df[popu_col[0]] + 1)
            df['geometry'] = df.apply(lambda row: Point(row[lon_col[0]], row[lat_col[0]], row[popu_col[0]]), axis=1)
            columns_to_keep = [lat_col[0], lon_col[0], popu_col[0], 'geometry']
        else:
            df['geometry'] = df.apply(lambda row: Point(row[lon_col[0]], row[lat_col[0]]), axis=1)
            columns_to_keep = [lat_col[0], lon_col[0], 'geometry']
        
        df = df[columns_to_keep]
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        return gdf
    else:
        raise ValueError("DataFrame Missing Columns 'lat' or 'lon'")

gdf1 = unifyDf(popuUrl)
gdf2 = unifyDf(stopsUrl)

x = gdf1.geometry.x
y = gdf1.geometry.y
weights = gdf1['U7H001']

models = {
    'Random Forest Test1': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Random Forest Test2': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    'Random Forest Test3': RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42),
    'Random Forest Test4': RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
}

nRange = range(50, 251, 5)
nRange1 = [115,140,170]
loopLog = []

for n in nRange1:
    hb = plt.hexbin(gdf1.geometry.x, gdf1.geometry.y, C=weights, gridsize=n, reduce_C_function=np.sum, cmap='viridis')

    hexbin_paths = hb.get_paths()
    hex_centroids = hb.get_offsets()

    vertices_list = []
    for path in hexbin_paths:
        if hasattr(path, 'vertices') and len(path.vertices) > 0:
            for v in path.vertices:
                for i in v:
                    if i != 0:
                        vertices_list.append(abs(v))

    medianPath = np.median(vertices_list)
    print(medianPath)

    hex_polygons = [Polygon(path.vertices) for path in hexbin_paths]
    hex_polygon_template = np.array(hex_polygons[0].exterior.coords)

    translated_hex_polygons = []
    for centroid in tqdm(hex_centroids, desc="Translating hexagons"):
        translated_vertices = hex_polygon_template + np.array(centroid)
        translated_hex_polygons.append(Polygon(translated_vertices))

    hex_centroids_points = [Point(c) for c in hex_centroids]
    hex_gdf = gpd.GeoDataFrame({'geometry': translated_hex_polygons, 'centroid': hex_centroids_points}, crs=gdf1.crs)

    gdf1_in_hex = gpd.sjoin(gdf1, hex_gdf, how="left", predicate="within")

    gdf2_in_hex = gpd.sjoin(gdf2, hex_gdf, how="left", predicate="within")

    population_sum = (
        gdf1_in_hex.groupby('index_right', group_keys=False)
        .progress_apply(lambda x: x['U7H001'].sum())
    ).reset_index(name='PopulationSum')

    bus_stops_count = (
        gdf2_in_hex.groupby('index_right', group_keys=False)
        .size()
    ).reset_index(name='BusStopsCount')

    hex_data = population_sum.merge(bus_stops_count, on='index_right', how='left').fillna(0)

    hex_data['HexbinCenter'] = hex_data['index_right'].apply(lambda idx: hex_centroids[idx])
    hex_data['HexbinCenter'] = hex_data['HexbinCenter'].apply(lambda coord: Point(coord))
    hex_gdf = gpd.GeoDataFrame(hex_data, geometry='HexbinCenter', crs="EPSG:4326")


    data = hex_data[['PopulationSum', 'BusStopsCount']]
    data['lon'] = hex_data['HexbinCenter'].apply(lambda point: point.x)
    data['lat'] = hex_data['HexbinCenter'].apply(lambda point: point.y)

    X = data[['PopulationSum', 'lon', 'lat']]
    y = data['BusStopsCount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_errors = {}

    for i, (name, model) in enumerate(tqdm(models.items(), desc="Training Models")):
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        score = r2_score(y_test, y_pred_test)
        colors_test = color_mapping(y_test, y_pred_test)

        errors = y_test - y_pred_test

        X_test_with_errors = X_test.copy()
        X_test_with_errors['Actual'] = y_test
        X_test_with_errors['Predicted'] = y_pred_test
        X_test_with_errors['Error'] = errors
        X_test_with_errors['radius'] = medianPath * np.cos(X_test_with_errors['lat'] * np.pi / 180) * 111320

        top_10_errors = X_test_with_errors.nsmallest(10, 'Error')

        loopLog.append((n, name, score))
        
print(loopLog)

looplog_df = pd.DataFrame(loopLog, columns=['n', 'Model', 'R² Score'])
plt.clf()
for model_name in looplog_df['Model'].unique():
    model_data = looplog_df[looplog_df['Model'] == model_name]
    plt.plot(model_data['n'], model_data['R² Score'], label=model_name)
print(looplog_df)
plt.xlabel('n')
plt.ylabel('R² Score')
plt.title('Model Performance Over Different Grid Sizes')
plt.legend()
plt.show()