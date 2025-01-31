import pandas as pd
import matplotlib.pyplot as plt

path = "D:/codes/hackru24fall/BusStop/Bus_stop/stops.txt"

df = pd.read_csv(path)  

pathstr = path.split('/')
for index, str in enumerate(pathstr):
    print(f"{index} : {str}")

print(pathstr[-1])
print(pathstr[:-1])
agencyurl = '/'.join(pathstr[:-1]) + '/agency.txt'

agencydf = pd.read_csv(agencyurl)

print(agencydf)

agencyname = agencydf['agency_name'][0]

print(agencyname)



# selected_columns = df[['stop_lat', 'stop_lon', 'stop_name']]

# print(selected_columns)

# plt.figure(figsize=(10, 6))

# plt.scatter(selected_columns['stop_lon'], selected_columns['stop_lat'], color='blue', marker='o', s=50)  # s为点的大小

# plt.title('Locations of Stops')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# plt.axis('equal')
# plt.grid()
# plt.show()
