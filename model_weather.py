import pandas as pd

# read merged weather dataset
#weather_data = pd.read_csv('datasets_cleaned/merged_weather_data.csv')
weather_data = pd.read_csv('datasets_cleaned/merged_weather_data_tiny.csv')

# get target and feature variables
target = 

feature_ls = list(weather_data)
feature_ls = feature_ls[:14] + feature_ls[18:20]
features = weather_data[feature_ls]