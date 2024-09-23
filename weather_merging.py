import pandas as pd

weather_data = pd.read_csv('cosc2669-or-cosc2186-WIL-project\datasets_cleaned\cleaned_weather_data.csv')
geo_data = pd.read_csv('SuburbClustered.csv')

print(weather_data.head())
print(geo_data.head())

#merge the datasets
merged_data = pd.merge(weather_data, geo_data, on=['ClusterID', 'ClusterID'], how='left')

print(merged_data.head())

merged_data.to_csv("merged_weather_data.csv", index=False)

