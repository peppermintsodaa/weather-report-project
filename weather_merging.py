import pandas as pd

weather_data = pd.read_csv('datasets_cleaned/cleaned_weather_data.csv')
geo_data = pd.read_csv('datasets/SuburbClustered.csv')

print(weather_data.head())
print(geo_data.head())

#merge the datasets
merged_data = pd.merge(weather_data, geo_data, on=['ClusterID', 'ClusterID'], how='left')

print(merged_data.head())
print(merged_data.info())

# WARNING: final dataset will be > 19 GB. please proceed with caution
#merged_data.to_csv("datasets_cleaned/merged_weather_data.csv", index=False)

# subsetted version of 250,000 randomly selected  entries to test model for now
merged_data_tiny = merged_data.sample(n = 250000, random_state = 69420)
merged_data_tiny.to_csv("datasets_cleaned/merged_weather_data_tiny.csv", index=False)