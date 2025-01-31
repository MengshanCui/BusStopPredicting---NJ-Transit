import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_file_paths(directory, filename):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                file_paths.append(os.path.join(root, file))
    return file_paths


selected_stops = ['(call to request service)']

def consolidate_data(directory, filename):
    file_paths = get_file_paths(directory, filename)
    data_frames = []
    

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # 检查 selected_stops 是否在 df 的 stop_id 列中
        # if 'stop_name' in df.columns:
        #     for stop in selected_stops:
        #         for index, row in df.iterrows():
        #             if stop in row['stop_name']:
        #                 print(row['stop_id'], row['stop_name'], file_path)
        data_frames.append(df)
    
    consolidated_df = pd.concat(data_frames, ignore_index=True)
    return consolidated_df

# Example usage
directory = 'BusStop'
stopname = 'stops.txt'
stoptimename = 'stop_times.txt'
stopdf = consolidate_data(directory, stopname)
timedf = consolidate_data(directory, stoptimename)


# # print(stopdf.head())
# print(stopdf.info())
# # print(timedf.head())
# print(timedf.info())



# 检查timedf是否有重复行
if stopdf.duplicated(subset=['stop_id']).any():
    print("stopdf contains duplicate rows.")
    duplicate_rows = stopdf[stopdf.duplicated(subset=['stop_id'])]
    print("Duplicate rows:")
    print(duplicate_rows)
    
    # 去除重复行
    stopdf = stopdf.drop_duplicates(subset=['stop_id'])
    print("DataFrame after removing duplicate rows:")
    # print(timedf)
else:
    print("stopdf does not contain duplicate rows.")

# 检查timedf是否有重复行
if timedf.duplicated().any():
    print("timedf contains duplicate rows.")
    duplicate_rows = timedf[timedf.duplicated()]
    print("Duplicate rows:")
    print(duplicate_rows)
    
    # 去除重复行
    timedf = timedf.drop_duplicates()
    print("DataFrame after removing duplicate rows:")
    # print(timedf)
else:
    print("timedf does not contain duplicate rows.")

stopdf.fillna(-1, inplace=True)
timedf.fillna(-1, inplace=True)
columns_to_describe = ['pickup_type', 'drop_off_type', 'arrival_time']  # 替换为你需要的列名
timedf[columns_to_describe].hist(figsize=(16, 15), bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.show()

if 'pickup_type' in timedf.columns:
    # 统计timedf中每个stop_id的pickup_type为1的数量
    pickup_type_1_count = timedf[timedf["pickup_type"] == 1.0 ].groupby('stop_id').size()
    # trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,continuous_pickup,continuous_drop_off,shape_dist_traveled,timepoint


    drop_off_type_1_count = timedf[timedf["drop_off_type"] == 1.0 ].groupby('stop_id').size()
    # trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_

    # 统计timedf中每个stop_id的总数据量
    total_count = timedf.groupby('stop_id').size()

    # 计算比例
    pickup_type_1_ratio = pickup_type_1_count / total_count
    drop_off_type_1_ratio = drop_off_type_1_count / total_count
    
    pickup_type_1_ratio.fillna(0, inplace=True)
    drop_off_type_1_ratio.fillna(0, inplace=True)

    total_ratio = pickup_type_1_ratio + drop_off_type_1_ratio

    # 将计算结果合并到stopdf中，新增一列
    stopdf = stopdf.set_index('stop_id')
    stopdf['pickup_type_1_ratio'] = pickup_type_1_ratio
    stopdf['drop_off_type_1_count'] = drop_off_type_1_ratio
    stopdf['total_ratio'] = total_ratio
    # 重置索引
    stopdf = stopdf.reset_index()
    
else:
    print("timedf does not contain 'pickup_type' column.")
    print(timedf.columns)


# 选择特定的列组成一个新的 DataFrame
selected_columns = ['stop_id', 'pickup_type_1_ratio', 'drop_off_type_1_count','total_ratio']  # 替换为你需要的列名
new_df = stopdf[selected_columns]


# 过滤掉 pickup_type_1_ratio 和 drop_off_type_1_count 都为 0 的数据
filtered_df = new_df[(new_df['pickup_type_1_ratio'] != 0) | (new_df['drop_off_type_1_count'] != 0)]


print(filtered_df.head())
print(filtered_df.shape)
print(filtered_df.describe())

# # 绘制直方图
# stopdf.hist(figsize=(16, 15), bins=20, color='blue', alpha=0.7, edgecolor='black')
# plt.show()

numeric_df = filtered_df.select_dtypes(include=[float, int])

# 计算相关矩阵
corr_matrix = numeric_df.corr()

# 使用 seaborn 绘制热力图
plt.figure(figsize=(16, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

for index, row in stopdf.iterrows():
    if row['total_ratio'] > 1:
        print(row['stop_id'],row['stop_name'], row['pickup_type_1_ratio'], row['drop_off_type_1_count'], row['total_ratio'])