import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("nhgis0001_ds258_2020_block.csv")

selected_columns = df[['INTPTLAT', 'INTPTLON', 'U7H001']]

print(selected_columns)

last_column_name = df.columns[-1]  
last_column_data = df[last_column_name]

# Calculate statistics
min_value = last_column_data.min()
q1 = last_column_data.quantile(0.25)
median = last_column_data.median()
q3 = last_column_data.quantile(0.75)
max_value = last_column_data.max()

print(f'Minimum value: {min_value}')
print(f'First quartile (Q1): {q1}')
print(f'Median (Q2): {median}')
print(f'Third quartile (Q3): {q3}')
print(f'Maximum value: {max_value}')

log_data = np.log(last_column_data + 1)

min_value = log_data.min()
q1 = log_data.quantile(0.25)
median = log_data.median()
q3 = log_data.quantile(0.75)
max_value = log_data.max()

print(f'Minimum value: {min_value}')
print(f'First quartile (Q1): {q1}')
print(f'Median (Q2): {median}')
print(f'Third quartile (Q3): {q3}')
print(f'Maximum value: {max_value}')

plt.figure(figsize=(10, 6))
plt.hist(log_data, bins=30, color='b', alpha=0.7)
plt.title('Distribution of Log-Transformed Data')
plt.xlabel('Log Values')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()
