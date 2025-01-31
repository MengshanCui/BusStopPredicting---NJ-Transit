import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import requests
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QSizePolicy, QFileDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
import gpxpy
import folium
from folium.plugins import HeatMap
import math


load_dotenv()

api_key = os.getenv("OPENROUTESERVICE_API_KEY")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.stopdf = None
        self.curdf = None

        self.setWindowTitle("Bus Stop Optimization Based on Population")
        self.setGeometry(100, 100, 800, 600)

        # for map in web
        self.browser = QWebEngineView()

        # create some SelectFile buttons
        self.buttonCsv = QPushButton("Select CSV File as Population")
        self.buttonCsv.clicked.connect(self.open_file_dialogCsv)

        self.buttonTxt = QPushButton("Select TXT File as Bus Stops")
        self.buttonTxt.clicked.connect(self.open_file_dialogTxt)

        self.buttonXlsx = QPushButton("Select XLSX File as Hospitals")
        self.buttonXlsx.clicked.connect(self.open_file_dialogXlsx)

        self.buttonGpx = QPushButton("Select GPX as Shopping Malls")
        self.buttonGpx.clicked.connect(self.open_file_dialogGpx)

        self.buttonHeatMap = QPushButton("Select CSV as HeatMap Input")
        self.buttonHeatMap.clicked.connect(self.open_file_dialogHeatMap)

        self.browser = QWebEngineView()
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buttonCsv)
        button_layout.addWidget(self.buttonTxt)
        button_layout.addWidget(self.buttonXlsx)
        button_layout.addWidget(self.buttonGpx)
        button_layout.addWidget(self.buttonHeatMap)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)  
        main_layout.addWidget(self.browser)    

        container = QWidget()
        container.setLayout(main_layout)

        self.setCentralWidget(container)

        # load HTML File
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_file_path = os.path.join(current_dir, "map.html")
        self.browser.setUrl(QUrl.fromLocalFile(html_file_path))

    def draw_route(self, start, end):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {'Authorization': api_key}
        params = {'start': f"{start[1]},{start[0]}", 'end': f"{end[1]},{end[0]}"}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            route_data = response.json()
            route_coords = route_data['features'][0]['geometry']['coordinates']
            js_code = f"drawRoute({route_coords});"
            self.browser.page().runJavaScript(js_code)
        else:
            print("Error fetching route:", response.json())

    def open_file_dialogCsv(self):
        # FileSelect
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            # bounds of lags
            self.browser.page().runJavaScript("getMapBounds();", lambda bounds: self.add_population_points(file_name, bounds))
    
    def open_file_dialogXlsx(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select XLSX File", "", "XLSX Files (*.xlsx);;All Files (*)", options=options)
        if file_name:
            self.browser.page().runJavaScript("getMapBounds();", lambda bounds: self.add_hospital_points(file_name, bounds))
    
    def open_file_dialogGpx(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select GPX File", "", "GPX Files (*.gpx);;All Files (*)", options=options)
        if file_name:
            self.browser.page().runJavaScript("getMapBounds();", lambda bounds: self.add_mall_points(file_name, bounds))

    def open_file_dialogTxt(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Txt File", "", "TXT Files (*.txt);;All Files (*)", options=options)
        if file_name:
            self.browser.page().runJavaScript("getMapBounds();", lambda bounds: self.add_stop_points(file_name, bounds))
    
    def open_file_dialogHeatMap(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Csv HeatMap File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.browser.page().runJavaScript("getMapBounds();", lambda bounds: self.add_heatMap_points(file_name, bounds))

    def add_hospital_points(self, csv_file, bounds):
        # Read Excel File
        df = pd.read_excel("Acute_Care_Hospitals_in_NJ.xlsx")

        # Select 
        selected_columns = df[['LATITUDE', 'LONGITUDE']]
        print(selected_columns.head())
        for index, row in selected_columns.iterrows():
            lat = row['LATITUDE']
            lon = row['LONGITUDE']
            
            radius = 150
            color = 'rgba(0, 0, 0)'  # white (but more like gray and black)
            
            self.add_marker(lat, lon, color, radius)

    def add_mall_points(self, csv_file, bounds):
        try:
            with open("nj-malls-and-shopping-centers.gpx", 'r', encoding='utf-8') as gpx_file:
                gpx = gpxpy.parse(gpx_file)

            points = []
            for waypoint in gpx.waypoints:
                points.append({'latitude': waypoint.latitude, 'longitude': waypoint.longitude})

            df = pd.DataFrame(points)

            print(df.head())

        except Exception as e:
            print(f"Error reading GPX file: {e}")
        
        for index, row in df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            radius = 150
            color = 'rgba(0, 255, 0)'  # green
            
            self.add_marker(lat, lon, color, radius)

    def add_stop_points(self, csv_file, bounds):
        if bounds:
            north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
            # Read CSV File
            df = pd.read_csv(csv_file)  

            print(f"Number of rows in the CSV file before: {df.shape[0]}")
            selected_columns = df[['stop_lat', 'stop_lon', 'stop_name']]

        # Sort 
        sorted_df = selected_columns.sort_values(by='stop_lon')

        # Thresholds for similarity 
        lat_threshold = 0.0005  # About 50 meters
        lon_threshold = 0.0005  

        grouped_points = []
        avg_lat = sorted_df.iloc[0]['stop_lat']
        avg_lon = sorted_df.iloc[0]['stop_lon']
        count = 1

        for i in range(1, len(sorted_df)):
            current_row = sorted_df.iloc[i]
            prev_row = sorted_df.iloc[i - 1]

            # Check if current point is similar to the previous one
            if (abs(current_row['stop_lat'] - prev_row['stop_lat']) < lat_threshold and
                abs(current_row['stop_lon'] - prev_row['stop_lon']) < lon_threshold):
                # Accumulate the latitude and longitude
                avg_lat += current_row['stop_lat']
                avg_lon += current_row['stop_lon']
                count += 1
            else:
                # Average the accumulated values
                new_row = {
                    'stop_lat': avg_lat / count,
                    'stop_lon': avg_lon / count,
                    'stop_name': prev_row['stop_name']  # Keep the name from the last point
                }
                grouped_points.append(new_row)

                # Reset for the next group
                avg_lat = current_row['stop_lat']
                avg_lon = current_row['stop_lon']
                count = 1

        # Handle the last group outside the loop
        if count > 0:
            new_row = {
                'stop_lat': avg_lat / count,
                'stop_lon': avg_lon / count,
                'stop_name': sorted_df.iloc[-1]['stop_name']  # Use the last name
            }
            grouped_points.append(new_row)

        # Convert to DataFrame for further processing
        grouped_df = pd.DataFrame(grouped_points)

        # test columns
        print(grouped_df.head())
        self.stopdf = pd.DataFrame(grouped_df)

        print(f"Number of rows in the CSV file after: {grouped_df.shape[0]}")
        # 绘制点
        for index, row in selected_columns.iterrows():
            lat = row['stop_lat']
            lon = row['stop_lon']
            
            # radius = 1609.34  # 1 mile（ 1609.34 meters）
            radius = 5
            color = 'rgba(0, 0, 255, 0.5)'  # half blue
            
            # # checking boundaries, only necessary for big sample.
            # if south <= lat <= north and west <= lon <= east:
            self.add_marker(lat, lon, color, radius)

    def add_population_points(self, csv_file, bounds):
        if bounds:
            north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
            
            # Read CSV File
            df = pd.read_csv(csv_file)
            

            # Select经纬度列和人口列
            selected_columns = df[['INTPTLAT', 'INTPTLON', 'U7H001']]
            
            # 对人口数进行对数变换
            selected_columns['U7H001'] = np.log(selected_columns['U7H001'] + 1)  # 加 1 以避免对数零的问题

            # 打印所选列的数据
            print(selected_columns.head())

            # 绘制点
            for index, row in selected_columns.iterrows():
                lat = row['INTPTLAT']
                lon = row['INTPTLON']
                log_pop = row['U7H001']
                radius = self.get_radius(log_pop)  # 根据对数人口数确定半径
                color = self.get_color(log_pop)  # 根据对数人口数确定颜色
                
                # 检查经纬度是否在地图边界内
                if south <= lat <= north and west <= lon <= east and log_pop >= 1:
                    self.add_marker(lat, lon, color, radius)

    def haversine(self, lat1, lon1, lat2, lon2):
        # Haversine formula to calculate the distance between two points on the Earth
        R = 6371  # Earth radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c  # returns distance in kilometers

    def add_heatMap_points(self, csv_file, bounds):
        if bounds:
            north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
            
            # Read CSV File
            df = pd.read_csv(csv_file)
            
            
            selected_columns = df[['INTPTLAT', 'INTPTLON', df.columns[-1]]]  # last col
            selected_columns = selected_columns.dropna()

            # np.log1p the population
            selected_columns[df.columns[-1]] = np.log1p(selected_columns[df.columns[-1]])

            # use selected_columns the first point as center
            first_point = selected_columns.iloc[0]
            location = [first_point['INTPTLAT'], first_point['INTPTLON']]

            m = folium.Map(location=location, zoom_start=5)
            HeatMap(data=selected_columns[['INTPTLAT', 'INTPTLON', df.columns[-1]]].values.tolist()).add_to(m)
            m.save("heatmap.html")

            # find top 20
            top_20 = selected_columns.nlargest(20, df.columns[-1])

            # find att back
            top_latitudes = top_20['INTPTLAT'].values
            top_longitudes = top_20['INTPTLON'].values
            top_local_max_values = top_20[df.columns[-1]].values

            # Paint the points
            for lat, lon, value in zip(top_latitudes, top_longitudes, top_local_max_values):
                radius = value * 100  
                color = 'yellow'  # obvious
                self.add_marker(lat, lon, color, radius)

                # if self.stopdf is not None and len(self.stopdf) > 0:
                #     shortest_distance = float('inf')
                #     closest_end_point = None

                #     # Calculate the closest endpoint using the Haversine formula
                #     for index, end in self.stopdf.iterrows():
                #         end_lat, end_lon = end["stop_lat"], end["stop_lon"]  # Adjust these to match your DataFrame columns
                #         # print(f"Checking endpoint {index}: {end_lat}, {end_lon}")
                        
                #         straight_distance = self.haversine(lat, lon, end_lat, end_lon)

                #         if straight_distance < shortest_distance:
                #             shortest_distance = straight_distance
                #             closest_end_point = (end_lat, end_lon)  # You can also store the entire row if needed
                    
                #     # Once we have the closest endpoint, call the OpenRouteService API
                #     if closest_end_point:
                #         print(f"Closest endpoint based on straight-line distance: {closest_end_point} with distance: {shortest_distance:.2f} km")

                #         # Call OpenRouteService API for the actual driving route
                #         url = "https://api.openrouteservice.org/v2/directions/driving-car"
                #         headers = {'Authorization': api_key}
                #         params = {'start': f"{lat},{lon}", 'end': f"{closest_end_point[0]},{closest_end_point[1]}"}

                #         response = requests.get(url, headers=headers, params=params)

                #         if response.status_code == 200:
                #             route_data = response.json()
                #             route_length = route_data['features'][0]['properties']['segments'][0]['distance']
                            
                #             print(f"Driving distance to closest endpoint: {route_length / 1000:.2f} km")
                #             # Draw the route on the map
                #             self.draw_route((lat, lon), closest_end_point)
                #         else:
                #             print("Error fetching route:", response.json())
                #     else:
                #         print("No valid endpoint found.")
    

    def get_color(self, log_population):
        # change color based on population
        if log_population < 1:
            return 'green' # no using
        elif log_population < 3:
            return 'pink'
        else:
            return 'red'

    def get_radius(self, log_population):
        # change radius based on population
        return max(5, log_population * 3)  # min = 5

    def add_marker(self, lat, lon, color, radius):
        js_code = f"addMarker({lat}, {lon}, '{color}', {radius});"
        self.browser.page().runJavaScript(js_code)

    def draw_route(self, start, end):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {'Authorization': api_key}
        params = {'start': f"{start[1]},{start[0]}", 'end': f"{end[1]},{end[0]}"}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            route_data = response.json()
            route_coords = route_data['features'][0]['geometry']['coordinates']
            js_code = f"drawRoute({route_coords});"
            self.browser.page().runJavaScript(js_code)
        else:
            print("Error fetching route:", response.json())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
