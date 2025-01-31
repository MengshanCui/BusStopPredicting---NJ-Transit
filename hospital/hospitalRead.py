import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Acute_Care_Hospitals_in_NJ.xlsx")

print(df)

selected_columns = df[['LATITUDE', 'LONGITUDE']]

print(selected_columns)

plt.figure(figsize=(10, 6))

plt.scatter(selected_columns['LONGITUDE'], selected_columns['LATITUDE'], color='blue', marker='o', s=50)  # s size

plt.title('Locations of Acute Care Hospitals in NJ')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.axis('equal')

plt.grid()

plt.show()
