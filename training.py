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
from scipy.spatial import cKDTree

import gpxpy

from sklearn.cluster import DBSCAN
# from osgeo import gdal
from tqdm import tqdm
tqdm.pandas()


def color_mapping(actual, predicted):
    residual = predicted - actual  
    norm = Normalize(vmin=residual.min(), vmax=residual.max() * 1.5)  # range
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(residual)  
    return colors



directory = '/Users/cuimengshan/Desktop/NJ transit/project-12.15 2/BusStop'
stopname = 'stops.txt'
popuUrl = "project-12.15 2/population/nhgis0001_ds258_2020_block.csv"
hosUrl = 'project-12.15 2/hospital/Acute_Care_Hospitals_in_NJ.xlsx'
malUrl = 'project-12.15 2/shoppingMall/nj-malls-and-shopping-centers.gpx'


def popuReader(fileurl):
    file_extension = os.path.splitext(fileurl)[1]
    if file_extension in ['.txt', '.csv']:
        df = pd.read_csv(fileurl, low_memory=False)
    else:
        print("Unsupported file format.")
        return None

    lat_col = [col for col in df.columns if 'lat' in col.lower()]
    lon_col = [col for col in df.columns if 'lon' in col.lower()]
    popu_col = [col for col in df.columns if 'u7h001' in col.lower()]

    if lat_col and lon_col and popu_col:

        logPopu_col = np.log(df[popu_col[0]] + 1)
        df['Log Population'] = logPopu_col
    
        new_df = df[[lat_col[0], lon_col[0], popu_col[0], 'Log Population']].copy()
        new_df.columns = ['Latitude', 'Longitude', 'Population', 'Log Population']
        new_df = new_df[new_df['Population'] >= 1].copy()
        # print(new_df.describe())
        return new_df
    else:
        print("Required columns not found in the DataFrame.")
        return None

def stopReader(directory, filename):
    stopdict = []
    tempdict = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                # file_paths.append(os.path.join(root, file))
                # print(root)
                # print(dirs)
                # print(file)
                stopurl = os.path.join(root, file)
                # print(stopurl)
                df = pd.read_csv(stopurl)

                stimeurl = root + '/stop_times.txt'
                stimedf = pd.read_csv(stimeurl)

                stimedf = stimedf[['stop_id', 'pickup_type', 'drop_off_type']].copy()
                
                # print(stimedf.describe())

                stimedf = stimedf.groupby('stop_id').agg(
                    totalPickUp=('pickup_type', lambda x: (x != 0).sum()), 
                    totalDropOff=('drop_off_type', lambda x: (x != 0).sum()), 
                    stopTimeCount=('stop_id', 'count')   
                ).reset_index()
                
                stimedf = stimedf[['stop_id', 'totalPickUp', 'totalDropOff', 'stopTimeCount']].copy()
                
                # print(stimedf.describe())

                df = pd.merge(df, stimedf, on='stop_id', how='inner')
                
                agencyurl = root + '/agency.txt'
                agdf = pd.read_csv(agencyurl)
                
                aglen = len(agdf)
                if aglen != 1:
                    agdf.loc[:, "agency_name"] = "Not Sure"
                    df['agency'] = agdf['agency_name'][0]    
                    tempdict.append(df)
                    
                else:
                    df['agency'] = agdf['agency_name'][0]    
                    # print(df.describe())
                    stopdict.append(df)
                # print(df['agency'])

    stopdf = pd.concat(stopdict, ignore_index=True)
    tempdf = pd.concat(tempdict, ignore_index=True)

    df = pd.concat([stopdf, tempdf], ignore_index=True)

    # print(df.describe())

    allstop = df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'agency', 'parent_station', 'totalPickUp', 'totalDropOff', 'stopTimeCount']].copy()
    allstop[['totalPickUp', 'totalDropOff', 'stopTimeCount']] = allstop[['totalPickUp', 'totalDropOff', 'stopTimeCount']].fillna(0).astype('int64')
    # print(allstop.describe())
    
    # nameDuplicates = allstop[allstop.duplicated(subset=['stop_name'], keep=False)]
    # geoDuplicates = allstop[allstop.duplicated(subset=['stop_lat','stop_lon'], keep=False)]

    # nameDuplicates = nameDuplicates.sort_values(by='stop_name', ascending=True)
    # nameDuplicates.to_csv('nameDuplicates.txt', sep=' ', index=False)

    # print(nameDuplicates.describe())
    # print(geoDuplicates.describe())
    
    # duplicatesA = nameDuplicates.sort_values(by='stop_name', ascending=True)
    # duplicatesA.to_csv('duplicates.txt', sep=' ', index=False)

    # print(allstop.columns)

    
    stopWithParent = allstop[allstop["parent_station"].notna()]

    
    for _, stop in stopWithParent.iterrows():
        parent_id = stop['parent_station']  # 获取父数据的 id
        allstop.loc[allstop['stop_id'] == parent_id, ['totalPickUp', 'totalDropOff', 'stopTimeCount']] += stop[['totalPickUp', 'totalDropOff', 'stopTimeCount']].astype('int64')
    
    mainstops = allstop[allstop["parent_station"].isna()]
    # allstop = allstop.duplicated(subset=['stop_name'])
    # allstop = allstop.drop_duplicates(subset=['stop_name'])

        # allstop['agency'] = allstop['agency'].fillna('').astype(str)
    # allstop['parent_station'] = allstop['parent_station'].fillna('').astype(str)

    # print(mainstops.describe())
    # Use .loc to modify the DataFrame without raising a warning
    mainstops = mainstops.copy()
    
    mainstops.loc[:, 'roundLat'] = mainstops['stop_lat'].round(3)
    mainstops.loc[:, 'roundLon'] = mainstops['stop_lon'].round(3)

    
    # by lat,lon
    merged_df = mainstops.groupby(['roundLat','roundLon'], as_index=False).agg({
        'stop_id': 'first',
        'stop_name': 'first',
        'stop_lat': 'first',
        'stop_lon': 'first',
        'totalPickUp': 'sum',
        'totalDropOff': 'sum',
        'stopTimeCount': 'sum',
        'agency': lambda x: ', '.join(set(x.dropna().astype(str)))  #merge
    })
    
    merged_df = merged_df.reset_index()

    merged_df['stop_name_lower'] = merged_df['stop_name'].str.lower()
    merged_df = merged_df.groupby('stop_name_lower', as_index=True).agg({
        'stop_id': 'first',
        'stop_name': 'first',
        'stop_lat': 'first',
        'stop_lon': 'first',
        'totalPickUp': 'sum',
        'totalDropOff': 'sum',
        'stopTimeCount': 'sum',
        'agency': lambda x: ', '.join(set(x.dropna().astype(str)))  # merge agency if different
    })

    threshold = 100  # metter
    # k = 3
    
    # # avg distances with k points (

    # coords = lat_lon_to_xy(merged_df['stop_lat'], merged_df['stop_lon'])
    # tree = cKDTree(coords)
    # distances, indices = tree.query(coords, k=k)  # k=2, 1 for itself
    # merged_df['nearestName'] = merged_df.iloc[indices[:, 1]]['stop_name'].values  
    # merged_df['nearestDistance'] = distances[:, 1]  
    # merged_df['avgDistances'] = distances[:, 1:].mean(axis=1)  
    # print(merged_df.describe())
    # # print(merged_df.head)

    
    # # nearNameDup = merged_df[merged_df.duplicated(subset=['nearest_point_name'])]
    # # print(nearNameDup.columns)
    
    # filteredDf = merged_df[merged_df['avgDistances'] < threshold]

    # oricol = merged_df.columns
    # oricol_list = list(oricol)
    # oricol_list.remove('stop_name')
    # merged_df = merged_df.merge(filteredDf, lefton='stop_name',righton='nearestName', how='left', suffixes=('', '_filtered'))
    
    # cols_to_update = {'totalPickUp', 'totalDropOff', 'stopTimeCount'}
    # print(merged_df.columns)
    # for col in cols_to_update:
    #     merged_df[col] += merged_df[f"{col}_filtered"].fillna(0)
        
    # merged_df = merged_df.drop(columns=[f"{col}_filtered" for col in oricol_list])
    # merged_df = merged_df[~df.isin(nearNameDup).all(axis=1)]

    stopgdf = merged_df.copy()
    stopgdf['geometry'] = merged_df.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
    
    stopgdf = gpd.GeoDataFrame(stopgdf, geometry='geometry', crs="EPSG:4326") 
    stopgdfp = stopgdf.to_crs("EPSG:32618")

    coords = np.array([stopgdfp.geometry.x, stopgdfp.geometry.y]).T
    
    db = DBSCAN(eps=threshold, min_samples=1).fit(coords)
    
    stopgdfp['cluster'] = db.labels_
    
    cluster_centers = stopgdfp.groupby('cluster').geometry.apply(lambda x: x.union_all().centroid)

    cluster_centers = cluster_centers.set_crs('EPSG:32618', allow_override=True).to_crs(epsg=4326) 
    
    aggregated_values = stopgdfp.groupby('cluster').agg({
        'totalPickUp': 'sum',
        'totalDropOff': 'sum',
        'stopTimeCount': 'sum',
    }).reset_index()
    
    result = pd.merge(cluster_centers, aggregated_values, on='cluster', how='left')
    result['stop_lon'] = result.geometry.x
    result['stop_lat'] = result.geometry.y


    coords = lat_lon_to_xy(result['stop_lat'], result['stop_lon'])
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=2)  # k=2, 1 for itself
    result['nearest_distance'] = distances[:, 1]  
    # print(result.head)

    # print(result.describe())
    result.to_csv('effectiveBusStop.csv', index=False) 

    return result

def lat_lon_to_xy(lat, lon):
    R = 6371000  # earth radius(meter
    x = R * np.radians(lon) * np.cos(np.radians(lat.mean()))
    y = R * np.radians(lat)
    return np.column_stack((x, y))  #meter

def hosReader(fileurl):
    file_extension = os.path.splitext(fileurl)[1]
    if file_extension in ['.xlsx']:
        df = pd.read_excel(fileurl)
    else:
        print("Unsupported file format.")
        return None
        
    new_df = df[['LATITUDE', 'LONGITUDE','LICENSED_NAME']]
    
    new_df = new_df.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon','LICENSED_NAME': 'name'})

    return new_df


def malReader(fileurl):
    file_extension = os.path.splitext(fileurl)[1]
    if file_extension in ['.gpx']:
        df = pd.read_csv(fileurl, low_memory=False)
    else:
        print("Unsupported file format.")
        return None

    try:
        with open(fileurl, 'r', encoding='utf-8') as gpx_file:
    
            gpx = gpxpy.parse(gpx_file)
    
            points = []
            for waypoint in gpx.waypoints:
                label = None
    
                for child in waypoint.extensions:
                # get the label_text element from .gpx file
                    label = child.find("{http://www.topografix.com/GPX/gpx_overlay/0/3}label_text")
                    if label is not None:
                        name = label.text
    
                points.append({
                    'lat': waypoint.latitude,
                    'lon': waypoint.longitude,
                    'name': name
                })
    
    
        df = pd.DataFrame(points)
    
        # print(df.head())
        return df
        # plt.show()
    
    except Exception as e:
        print(f"Error reading GPX file: {e}")
        return None

pdf = popuReader(popuUrl)
# print(pdf.describe())
stopdf = stopReader(directory, stopname)
hosdf = hosReader(hosUrl)
# print(hosdf.describe())
maldf = malReader(malUrl)
# print(maldf.describe())
# print(stopdf.head)
# stopdf.to_csv('stopdf.txt', sep=' ', index=False)
# print(stopdf.head)

fig, axs = plt.subplots(2, 5, figsize=(18, 12))
axs = axs.flatten()

popgdf = pdf.copy()

popgdf['geometry'] = pdf.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)

popgdf = gpd.GeoDataFrame(popgdf, geometry='geometry', crs="EPSG:4326") 

popgdfp = popgdf.to_crs("EPSG:32618")


# print(popgdf.columns)
# print(gdf.columns)

stopgdf = stopdf.copy()
stopgdf['geometry'] = stopdf.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)

stopgdf = gpd.GeoDataFrame(stopgdf, geometry='geometry', crs="EPSG:4326") 
stopgdfp = stopgdf.to_crs("EPSG:32618")


hosgdf = hosdf.copy()
hosgdf['geometry'] = hosdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

hosgdf = gpd.GeoDataFrame(hosgdf, geometry='geometry', crs="EPSG:4326") 
hosgdfp = hosgdf.to_crs("EPSG:32618")


malgdf = maldf.copy()
malgdf['geometry'] = maldf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

malgdf = gpd.GeoDataFrame(malgdf, geometry='geometry', crs="EPSG:4326") 
malgdfp = malgdf.to_crs("EPSG:32618")

# print(stopgdf.columns)

# Hexbin 
hb = plt.hexbin(popgdfp.geometry.x, popgdfp.geometry.y, C=popgdfp['Log Population'], gridsize=140, reduce_C_function=np.sum, cmap='viridis')
# plt.close() 

# lonMax = g

# center and paths(vertices)
hexbin_paths = hb.get_paths()
hex_centroids = hb.get_offsets()

# hexbin_paths, len is 1, in case. 
hex_polygons = [Polygon(path.vertices) for path in hexbin_paths]
hex_polygon_template = np.array(hex_polygons[0].exterior.coords)
print(hex_polygon_template)

# hex_polygons after translation
translated_hex_polygons = []
for centroid in tqdm(hex_centroids, desc="Translating hexagons"):
    translated_vertices = hex_polygon_template + np.array(centroid)
    translated_hex_polygons.append(Polygon(translated_vertices))
    
# print(translated_hex_polygons[0])

# hex_centroids_points = [Point(c) for c in hex_centroids]
hex_gdf = gpd.GeoDataFrame({'geometry': translated_hex_polygons}, crs=popgdfp.crs)

# connect hexagons and data points
popgdf_in_hex = gpd.sjoin(popgdfp, hex_gdf, how="left", predicate="within")
stopgdf_in_hex = gpd.sjoin(stopgdfp, hex_gdf, how="left", predicate="within")
hosgdf_in_hex = gpd.sjoin(hosgdfp, hex_gdf, how="left", predicate="within")
malgdf_in_hex = gpd.sjoin(malgdfp, hex_gdf, how="left", predicate="within")

# print(popgdfp)
# print(hex_gdf)
# print(popgdf_in_hex.columns)
# print(stopgdf_in_hex.columns)

# population sum
population_sum = (
    popgdf_in_hex.groupby('index_right', group_keys=False)
    .progress_apply(lambda x: x['Population'].sum())
).reset_index(name='PopulationSum')

# bus stops sum
# Index(['index', 'stop_name_lower', 'stop_id', 'stop_name', 'stop_lat',
       # 'stop_lon', 'totalPickUp', 'totalDropOff', 'stopTimeCount', 'agency',
       # 'geometry', 'index_right', 'centroid'],
bus_stops_count = (
    stopgdf_in_hex.groupby('index_right', group_keys=False)
    .progress_apply(lambda x: pd.Series({
        'busStopCount': x.shape[0],
        'pickUpSum': x['totalPickUp'].sum(),
        'dropOffSum': x['totalDropOff'].sum(),
        'stopTimeSum': x['stopTimeCount'].sum(),
    }))).reset_index(names='index_right')


# population sum
hospital_sum = (
    hosgdf_in_hex.groupby('index_right', group_keys=False)
    .size()
).reset_index(name='HospitalSum')


# population sum
mall_sum = (
    malgdf_in_hex.groupby('index_right', group_keys=False)
    .size()
).reset_index(name='ShoppingMalSum')


# popDuplicates = population_sum[population_sum.duplicated(subset=['index_right'], keep=False)]
# print(popDuplicates)
# print(bus_stops_count.columns)
# busDuplicates = bus_stops_count[bus_stops_count.duplicated(subset=['index_right'], keep=False)]
# print(busDuplicates)

hex_data = population_sum.merge(bus_stops_count, on='index_right', how='left').fillna(0)
hex_data = hex_data.merge(hospital_sum, on='index_right', how='left').fillna(0)
hex_data = hex_data.merge(mall_sum, on='index_right', how='left').fillna(0)

#     # nameDuplicates = allstop[allstop.duplicated(subset=['stop_name'], keep=False)]
# popDuplicates = population_sum[population_sum.duplicated(subset=['index_right'], keep=False)]
# busDuplicates = bus_stops_count[bus_stops_count.duplicated(subset=['index_right'], keep=False)]

# print(hex_data)

# print(hex_gdf.crs)

hex_gdf['area_m2'] = hex_gdf['geometry'].area 

hex_gdf['radius(miles)'] = np.sqrt(hex_gdf['area_m2'] / np.pi / 1e6) * 0.621371

hex_gdf['radius(meters)'] = np.sqrt(hex_gdf['area_m2'] / np.pi)


merged_gdf = hex_gdf.merge(
    hex_data,
    left_index=True,   
    right_on='index_right', 
    how='left'       
)


# hex_gdf['radius（by Meadian)'] = medianPath

# print(hex_gdf.describe())

merged_gdf['centroid'] = merged_gdf.geometry.centroid
merged_gdf = merged_gdf.to_crs("EPSG:4326")

centroid_gseries = gpd.GeoSeries(merged_gdf['centroid'], crs="EPSG:32618").to_crs(epsg=4326)
merged_gdf['centroid'] = centroid_gseries

# print(merged_gdf.head)


# print(merged_gdf)


data = merged_gdf[['busStopCount',
                 'PopulationSum', 
                 'HospitalSum',
                 'ShoppingMalSum',
                 'pickUpSum',
                 'dropOffSum',
                 'stopTimeSum',
                  'radius(miles)',
                   'radius(meters)',
                   'centroid',]]
data.loc[:,'lon'] = data.loc[:,'centroid'].x
data.loc[:,'lat'] = data.loc[:,'centroid'].y

# print(data.head)

# save data for display
data.to_csv('hex_dataForTrain.csv', index=False) 

x = data[['PopulationSum', 
          'stopTimeSum']]
y = data['busStopCount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

with pd.ExcelWriter('modelData.xlsx') as writer:
    x_train.to_excel(writer, sheet_name='x_train', index=False)
    x_test.to_excel(writer, sheet_name='x_test', index=False)
    y_train.to_excel(writer, sheet_name='y_train', index=False)
    y_test.to_excel(writer, sheet_name='y_test', index=False)


# find some models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Support Vector Regression': SVR(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
}

# possible hexbin for new bus stops
model_errors = {}

for i, (name, model) in enumerate(tqdm(models.items(), desc="Training Models")):
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    # y_test_pred = pd.DataFrame(y_test_pred, columns=['Prediction'])

    colors_test = color_mapping(y_test, y_test_pred)
    y_test_pred = pd.Series(y_test_pred, name='Prediction')
    score = r2_score(y_test, y_test_pred)

    errors = y_test - y_test_pred

    y_all_pred = model.predict(x)
    y_all_pred = pd.Series(y_all_pred, name='Prediction')
    errors_all = y - y_all_pred


    # add errors to data
    # 使用两个列进行 merge 过滤
    # x_test_keys = x_test[['lat', 'lon']]
    # x_test_keys['Actual'] = y_test.round(2)


    # print(isinstance(x_test, pd.DataFrame))
    # print(isinstance(y_test, pd.DataFrame))
    # print(isinstance(y_test_pred, pd.DataFrame))
    # print(isinstance(errors, pd.DataFrame))
    result = pd.concat([y, y_all_pred, errors_all], axis=1) 
    result = result.set_axis(['ActualBusStop', 'PredictedBusStop', 'Error'], axis=1)  # 设置列名
    result.loc[:, 'ActualBusStop'] = result.loc[:, 'ActualBusStop'].round(2)
    # x_test_keys['Predicted'] = y_test_pred.round(2)
    result.loc[:, 'PredictedBusStop'] = result.loc[:, 'PredictedBusStop'].round(2)
    # x_test_keys['Error'] = errors.round(2)
    result.loc[:, 'Error'] = result.loc[:, 'Error'].round(2)

    result = pd.merge(result, data, left_index=True, right_index=True, how='left')

    # print(result.columns)
    result.drop(columns=['busStopCount'], inplace=True)

    result['lon'] = result['lon'].round(3)
    result['lat'] = result['lat'].round(3)
    result['PopulationSum'] = result['PopulationSum'].round(3)
    
    # x_test_with_errors['Actual'] = y_test.round(2)
    # x_test_with_errors['Predicted'] = y_test_pred.round(2)
    # x_test_with_errors['Error'] = errors.round(2)
    result['radius(miles)'] = result['radius(miles)'].round(2)
    result['radius(meters)'] = result['radius(meters)'].round(2) 
    

    # # 1. 复制 DataFrame
    # df_copy = df.copy()

    # # 2. 按列 'B' 降序排列
    # df_copy = df_copy.sort_values(by='B', ascending=False)

    # # 3. 选择列 'C' > 0 的数据
    # df_filtered = df_copy[df_copy['C'] > 0]
    
    # top_10_errors = result.nsmallest(10, 'Error')
    # # bus_top_10_hexbin = result.nlargest(10, 'Actual')

    
    # bus_top_10_hexbin.to_csv(f"{name} bus prediction.txt", sep="\t", index=False)

    result = result[result['ActualBusStop']>0]

    model_errors[name] = result

    # first row
    axs[i].scatter(x_test['PopulationSum'], y_test, c=colors_test, s=50)
    axs[i].set_title(f"{name} (R² Score: {score:.2f})")
    axs[i].set_xlabel("Population")
    axs[i].set_ylabel("Actual Bus Stops")
    
    # second row
    y_pred_data = model.predict(data[['PopulationSum', 
          'stopTimeSum']])
    colors_data = color_mapping(data['busStopCount'], y_pred_data)
    axs[i + 5].scatter(data['lon'], data['lat'], c=colors_data, s=2)
    axs[i + 5].set_title(f"{name} Prediction on Data")
    axs[i + 5].set_xlabel("lon")
    axs[i + 5].set_ylabel("lat")

    
# possible points for new bus stops
for name, df in model_errors.items():
    df.to_csv(f'model_errors_{name}.csv', index=False)

# save models for display
with open('multiple_models.pkl', 'wb') as file:
    pickle.dump(models, file)
# print(model_errors)


# all data
axs[4].scatter(data['PopulationSum'], data['busStopCount'], c='blue', s=50)
axs[4].set_title("All Original Data")
axs[4].set_xlabel("Population")
axs[4].set_ylabel("Actual Bus Stops")
    

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.4)  


for ax in axs:
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.show()
