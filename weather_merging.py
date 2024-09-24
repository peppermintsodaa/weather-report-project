import pandas as pd

weather_data = pd.read_csv('datasets_cleaned\cleaned_weather_data.csv')
geo_data = pd.read_csv('weather_datasets\SuburbClustered.csv')

print(weather_data.head())
print(geo_data.head())

weather_data = geo_data.drop(columns=['GeoShape'])

#merge the datasets
merged_data = pd.merge(weather_data, geo_data, on=['ClusterID', 'ClusterID'], how='left')

print(merged_data.head())

merged_data.to_csv("datasets_cleaned\merged_weather_data.csv", index=False)

