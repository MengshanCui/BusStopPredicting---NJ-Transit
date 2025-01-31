import gpxpy
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


try:
    with open("nj-malls-and-shopping-centers.gpx", 'r', encoding='utf-8') as gpx_file:

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

    print(df.head())


    plt.figure(figsize=(10, 6))

    plt.scatter(df['lat'], df['lon'], color='blue', marker='o', s=50)  

    plt.title('Locations of Acute Care Hospitals in NJ')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.axis('equal')

    plt.grid()

    # plt.show()

except Exception as e:
    print(f"Error reading GPX file: {e}")