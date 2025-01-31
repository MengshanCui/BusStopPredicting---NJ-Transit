import os
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QWidget, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pages.file_dialogs import open_file_dialogCsv, open_file_dialogTxt, open_file_dialogXlsx, open_file_dialogGpx, open_file_dialogHeatMap, open_file_dialogTrainedData, open_file_dialogShape, alls_bounds, open_file_dialogBoundedTrainedData
import pandas as pd
import numpy as np
import gpxpy
from folium.plugins import HeatMap
import math
import json
import colorsys
import random
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt





class mapPage(QWidget):
    def __init__(self, pltPage):
        super().__init__()

        self.browser = QWebEngineView()
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        buttonCsv = QPushButton("Select CSV File as Population")
        buttonCsv.clicked.connect(lambda: open_file_dialogCsv(self))

        buttonClrPo = QPushButton("Clear Point")
        buttonClrPo.clicked.connect(lambda: self.clear_point())

        buttonAllS = QPushButton("Load All Stops")
        buttonAllS.clicked.connect(lambda: alls_bounds(self))

        buttonClrSh = QPushButton("Clear Shape")
        buttonClrSh.clicked.connect(lambda: self.clear_line())

        buttonTxt = QPushButton("Select TXT File as Bus Stops")
        buttonTxt.clicked.connect(lambda: open_file_dialogTxt(self))

        buttonXlsx = QPushButton("Select XLSX File as Hospitals")
        buttonXlsx.clicked.connect(lambda: open_file_dialogXlsx(self))

        buttonGpx = QPushButton("Select GPX as Shopping Malls")
        buttonGpx.clicked.connect(lambda: open_file_dialogGpx(self))

        buttonTrainedData = QPushButton("Select CSV as TrainedData Input(Top 10)")
        buttonTrainedData.clicked.connect(lambda: open_file_dialogTrainedData(self))

        buttonBoundedTrainedData = QPushButton("Select CSV as TrainedData Input(Bounded)")
        buttonBoundedTrainedData.clicked.connect(lambda: open_file_dialogBoundedTrainedData(self))

        button_layout = QHBoxLayout()
        button_layout.addWidget(buttonCsv)
        button_layout.addWidget(buttonClrPo)
        button_layout.addWidget(buttonAllS)
        button_layout.addWidget(buttonClrSh)
        button_layout.addWidget(buttonTxt)
        button_layout.addWidget(buttonXlsx)
        button_layout.addWidget(buttonGpx)
        button_layout.addWidget(buttonTrainedData)
        button_layout.addWidget(buttonBoundedTrainedData)

        search_layout = QHBoxLayout()
        self.longitude_input = QLineEdit()
        self.longitude_input.setPlaceholderText("Longitude")
        self.latitude_input = QLineEdit()
        self.latitude_input.setPlaceholderText("Latitude")
        search_button = QPushButton("Go")
        search_button.clicked.connect(self.goButoon)
        self.dropdown = QComboBox()
        # Width control for the dropdown
        self.dropdown.addItem(" ---------------------- Predicted Locations ---------------------- ")  # Add an empty item to theself.dropdown

        search_layout.addWidget(self.longitude_input)
        search_layout.addWidget(self.latitude_input)
        search_layout.addWidget(search_button)
        search_layout.addWidget(self.dropdown)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.browser)
        main_layout.addLayout(search_layout)
    
        self.setLayout(main_layout)

        # Connect theself.dropdown signal to the slot
        self.dropdown.currentIndexChanged.connect(self.update_inputs_from_dropdown)

        # load HTML File
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_file_path = os.path.join(current_dir, "map.html")
        self.browser.setUrl(QUrl.fromLocalFile(html_file_path))

        self.pltPage = pltPage

    def add_marker(self, lat, lon, color, radius, info):
        info_json = json.dumps(info)
        js_code = f"addMarker({lat}, {lon}, '{color}', {radius}, {info_json});"
        self.browser.page().runJavaScript(js_code)

    def add_line(self, group, color):
        js_code = f"addLine({group}, '{color}');"
        self.browser.page().runJavaScript(js_code)

    def clear_point(self):
        js_code = f"clearPoint();"
        self.browser.page().runJavaScript(js_code)

    def clear_line(self):
        js_code = f"clearLine();"
        self.browser.page().runJavaScript(js_code)

    def goButoon(self):
        if self.longitude_input.text().strip() and self.latitude_input.text().strip():
            self.mapSetView(
                float(self.longitude_input.text()), 
                float(self.latitude_input.text()), 
                13
            )
    #     # 使用 QTimer 延时执行 runJavaScript
    #     QTimer.singleShot(0, self.updataPlot)
        
    # def updataPlot(self):
    #     self.browser.page().runJavaScript("getMapBounds();", 
    #         lambda bounds: self.pltPage.update_plot(bounds)
    #     )

    
    def generate_balanced_colors(self, n):
        colors = []
        for i in range(n):
            hue = 0.5 + (i / n) * 0.25 + random.uniform(-0.05, 0.05)  # Random change
            lightness = 0.5 + random.uniform(0, 0.4)  
            saturation = 0.5 + random.uniform(-0.4, 0.4)  
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors
    
    def agencyColor(self, seed):
        random.seed(seed)
        hue = 0.5 + random.uniform(-0.05, 0.05)  # Random change
        lightness = 0.5 + random.uniform(0, 0.4)  
        saturation = 0.5 + random.uniform(-0.4, 0.4)  
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        return hex_color

    # def draw_route(self, start, end):
    #     draw_route(self, start, end)

    def add_shape_points(self, agency_name, csv_file, bounds):
        try:
            if bounds:
                north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
                # Read CSV File
                df = pd.read_csv(csv_file)

                shape_dict = {}
                # manipulate by shape_id
                grouped = df.groupby('shape_id')

                filteredDf = df[(df['shape_pt_lat'] >= south) & (df['shape_pt_lat'] <= north) & (df['shape_pt_lon'] >= west) & (df['shape_pt_lon'] <= east)]

                color = self.agencyColor(agency_name)
                # print(f"Agency: {agency_name}, color: {color}")
                # self.clear_line()
              
                for shape_id, group in grouped:
                    if shape_id not in filteredDf['shape_id'].values:
                        continue
                    lat_lon_list = list(zip(group['shape_pt_lat'], group['shape_pt_lon']))
                    shape_dict[shape_id] = lat_lon_list
                    info = f"shape_id: {shape_id}"

                    
                    # print(f" {shape_id} : {len(lat_lon_list)}")

                    # print(lat_lon_list[0])
                    
                    # print(color)
                    js_latlngs = '[' + ', '.join(f'[{lat}, {lon}]' for lat, lon in lat_lon_list) + ']'
                    self.add_line(js_latlngs, color)
        except Exception as e:
            print(f"Error reading Shape file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading Shape file: {e}")

    def add_hospital_points(self, csv_file, bounds):
        # Read Excel File
        try:
            df = pd.read_excel(csv_file)

            # Select 
            # print(df.head())
            for index, row in df.iterrows():
                lat = row['LATITUDE']
                lon = row['LONGITUDE']
                name = row['LICENSED_NAME']
                
                radius = 150
                color = 'rgba(0, 0, 0)'  # white 
                info = name
                self.add_marker(lat, lon, color, radius, info)

        except Exception as e:
            print(f"Error reading Excel file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading Excel file: {e}")

    def add_mall_points(self, csv_file, bounds):
        try:
            with open(csv_file, 'r', encoding='utf-8') as gpx_file:
                gpx = gpxpy.parse(gpx_file)

                points = []
                for waypoint in gpx.waypoints:

                    label = None
                    name = "Shopping Mall Name Missed"
                    for child in waypoint.extensions:
                    # A way to read label_text in .gpx file
                        label = child.find("{http://www.topografix.com/GPX/gpx_overlay/0/3}label_text")
                        if label is not None:
                            name = label.text

                    points.append({
                        'lat': waypoint.latitude,
                        'lon': waypoint.longitude,
                        'name': name
                    })
                    # print(points)

            # print(points)
            df = pd.DataFrame(points)

            # print(df.head())

            for index, row in df.iterrows():
                lat = row['lat']
                lon = row['lon']
                name = row['name']
                radius = 150
                color = 'rgba(0, 255, 0)'  # green
                info = name
                self.add_marker(lat, lon, color, radius, info)
                print(f"Shopping Mall Name: {name}")

        except Exception as e:
            print(f"Error reading GPX file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading GPX file: {e}")

    def add_all_stops(self, bounds):
        try:
            if bounds:
                north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
                
                busfolders = '/Users/cuimengshan/Desktop/NJ transit/project-12.15 2/BusStop'
                total_folders = len(os.listdir(busfolders))

                progress_dialog = QProgressDialog("Loading stops...", "Cancel", 0, total_folders, self)
                progress_dialog.setWindowTitle("Progress")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setValue(0)

                for i, busfolder in enumerate(os.listdir(busfolders)):
                    if progress_dialog.wasCanceled():
                        break
                    stopurl = f"{busfolders}/{busfolder}/doesntmatter.txt"
                    self.add_stop_points(stopurl, bounds)
                    progress = int((i + 1) / total_folders * 100)
                    progress_dialog.setValue(i + 1)
                    QApplication.processEvents()
                # print(f"{busfolders}/{busfolder}/doesntmatter.txt")

        except Exception as e:
            print(f"Error reading file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading file: {e}")

    def add_stop_points(self, csv_file, bounds):
        try:
            if bounds:
                north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
                # Read CSV File
                csv_file = csv_file.split('/')
                agencyurl = '/'.join(csv_file[:-1]) + '/agency.txt'
                stopsurl = '/'.join(csv_file[:-1]) + '/stops.txt'
                shapeurl = '/'.join(csv_file[:-1]) + '/shapes.txt'

                
                agdf = pd.read_csv(agencyurl)
                agencyname = ""

                aglen = len(agdf)
                if aglen != 1:
                    agencyname = "Not Sure"
                    # tempdict.append(df)
                    
                else:
                    agencyname = agdf['agency_name'][0]    
                    # print(df.describe())
                    # stopdict.append(df)
                
                # stop
                stdf = pd.read_csv(stopsurl) 

                # shape
                self.add_shape_points(agencyname, shapeurl, bounds)

                # print(f"Number of rows in the CSV file before: {stdf.shape[0]}")
                selected_columns = stdf[['stop_lat', 'stop_lon', 'stop_name']]

            for index, row in selected_columns.iterrows():
                lat = row['stop_lat']
                lon = row['stop_lon']
                
                # radius = 1609.34  # 1 mile（ 1609.34 meters）
                radius = 5
                color = self.agencyColor(agencyname)
                info = f"Agency: {agencyname} <br> Stop Name: {row['stop_name']}"
                # # checking boundaries, only necessary for big sample.
                # if south <= lat <= north and west <= lon <= east:
                self.add_marker(lat, lon, color, radius, info)

        except Exception as e:
            print(f"Error reading TXT file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading TXT file: {e}")

    def add_population_points(self, csv_file, bounds):
        try:
            if bounds:
                north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
                
                # Read CSV File
                df = pd.read_csv(csv_file)
                
                # Select lat, lon, and population
                selected_columns = df[['INTPTLAT', 'INTPTLON', 'U7H001']]
                selected_columns['LogPo'] = np.log(selected_columns['U7H001'] + 1)  # log transformation

                # print(selected_columns.head())

                # addpoints
                for index, row in selected_columns.iterrows():
                    lat = row['INTPTLAT']
                    lon = row['INTPTLON']
                    log_pop = row['LogPo']
                    radius = self.get_radius(log_pop)  # change radius based on population
                    color = self.get_color(log_pop)  # change color based on population
                    info = f"Population: {row['U7H001']}" 
                    
                    # display area
                    if south <= lat <= north and west <= lon <= east and log_pop >= 1:
                        self.add_marker(lat, lon, color, radius, info)
                        
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            QMessageBox.warning(self, "Error", f"Error reading CSV file: {e}")
    
    def add_trained_points(self, csv_file, bounds):

        # Read CSV File
        df = pd.read_csv(csv_file)
        # print(df.head)
        # addpoints
        df = df.nsmallest(10, 'Error')

        for index, row in df.iterrows():
            lat = row['lat']
            lon = row['lon']
            mRadius = row['radius(meters)']
            mileRadius = row['radius(miles)']
            color = 'yellow'
            info = f"""Predicted Area for A New Bus Stops: <br>Actual Bus Stop Number: {row['ActualBusStop']}, <br>Predicted BusStop: {row['PredictedBusStop']}, <br>
                Error: {row['Error']}, <br> Radius: {mileRadius} miles （{mRadius} meters） <br> Population: {row['PopulationSum']}, Stoptime: {row['stopTimeSum']}"""
            self.add_marker(lat, lon, color, mRadius, info)

            self.dropdown.addItem(f"{lon}, {lat},{row['ActualBusStop']}, {row['PredictedBusStop']}, {row['Error']}")

    def add_bounded_trained_point(self, csv_file, bounds):

        # Read CSV File
        df = pd.read_csv(csv_file)
        # print(df.head)
        # addpoints
        # print(bounds)

        filtered_df = df[
            (df['lon'] >= bounds['west']) &
            (df['lon'] <= bounds['east']) &
            (df['lat'] >= bounds['south']) &
            (df['lat'] <= bounds['north'])
        ]

        df = filtered_df.nsmallest(3,'Error')

        self.dropdown.clear()
        for index, row in df.iterrows():
            lat = row['lat']
            lon = row['lon']
            mRadius = row['radius(meters)']
            mileRadius = row['radius(miles)']
            color = 'yellow'
            info = f"""Predicted Area for A New Bus Stops: <br>Actual Bus Stop Number: {row['ActualBusStop']}, <br>Predicted BusStop: {row['PredictedBusStop']}, <br>
                Error: {row['Error']}, <br> Radius: {mileRadius} miles （{mRadius} meters） <br> Population: {row['PopulationSum']}, Stoptime: {row['stopTimeSum']}"""
            self.add_marker(lat, lon, color, mRadius, info)

            self.dropdown.addItem(f"{lon}, {lat},{row['ActualBusStop']}, {row['PredictedBusStop']}, {row['Error']}")

    def update_inputs_from_dropdown(self):  
        selected_text =self.dropdown.currentText()
        if selected_text and selected_text != " -- Predicted Locations -- ":
            print(selected_text)
            lon, lat, Actual, Predicted, Error = [item.strip() for item in selected_text.split(",")]
            self.latitude_input.setText(lat)
            self.longitude_input.setText(lon)

    def haversine(self, lat1, lon1, lat2, lon2):
        # Haversine formula to calculate the distance between two points on the Earth
        R = 6371  # Earth radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c  # returns distance in kilometers
    
    def get_radius(self, log_population):
        # change radius based on population
        return max(5, log_population * 3)  # min = 5
    
    def get_color(self, log_population):
        # change color based on population
        if log_population < 1:
            return 'green' # no using
        elif log_population < 3:
            return 'pink'
        else:
            return 'red'
    
    def mapSetView(self, lon, lat, zoom):
        js_code = f"map.setView([{lat}, {lon}], {zoom});"
        print(lon, lat, zoom)
        self.browser.page().runJavaScript(js_code)