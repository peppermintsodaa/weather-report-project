import pandas as pd
import numpy as np

weather_data = pd.read_csv('datasets/WeatherData.csv')

print(weather_data.head())

#convert to datetime format
weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'], errors='coerce')

#take the data from the last 5 years
weather_data = weather_data[weather_data['Datetime'].dt.year >= 2019]

#going to drop the columns with too many missing values
weather_data = weather_data.loc[:, weather_data.isnull().mean() < 0.5] 

print(weather_data.head()) #apparently there's none, so that's good
print(weather_data.info())

#fill missing values with their mean value
numeric_columns = weather_data.select_dtypes(include=[np.float64, np.int64]).columns
weather_data[numeric_columns] = weather_data[numeric_columns].fillna(weather_data[numeric_columns].mean())

print(weather_data.head())

#seperate the year, month, day, and hour
weather_data['Year'] = weather_data['Datetime'].dt.year
weather_data['Month'] = weather_data['Datetime'].dt.month
weather_data['Day'] = weather_data['Datetime'].dt.day
weather_data['Hour'] = weather_data['Datetime'].dt.hour

print(weather_data.head())

#drop the datetime column 
weather_data = weather_data.drop(columns=['Datetime'])

print(weather_data.info())

#merge the coordinates to the cluster id
clusters_data = pd.read_csv('datasets/Clusters.csv')
weather_data = pd.merge(weather_data, clusters_data, on='ClusterID', how='left')

#save to csv
weather_data.to_csv("datasets_cleaned/cleaned_weather_data.csv", index=False)