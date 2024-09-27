import pandas as pd

# Load datasets
weather_data = pd.read_csv('datasets_cleaned/cleaned_weather_data.csv')
clusters = pd.read_csv('datasets/Clusters.csv')
suburb_clustered = pd.read_csv('datasets/SuburbClustered.csv')

# Clean and preprocess the data
weather_data.fillna(method='ffill', inplace=True)

# Merge weather_data with clusters to get latitude and longitude
weather_data = weather_data.merge(clusters[['ClusterID', 'Latitude', 'Longitude']], on='ClusterID')

# Merge with suburb_clustered to get suburb information
weather_data = weather_data.merge(suburb_clustered[['ClusterID', 'OfficialNameSuburb', 'OfficialNameState']], on='ClusterID')

# Group data by suburb and month
grouped = weather_data.groupby(['OfficialNameSuburb', 'Month']).agg({
    'TemperatureMean': 'mean',          # Average temperature
    'RainSum': 'sum',                   # Total rainfall
    'RelativeHumidityMean': 'mean'      # Average humidity
}).reset_index()

# Calculate chances of disasters
def calculate_disasters(row):
    temp = row['TemperatureMean']
    rainfall = row['RainSum']
    humidity = row['RelativeHumidityMean']
    
    drought_chance = 0
    flood_chance = 0
    bushfire_chance = 0

    # Define criteria for each disaster
    if rainfall < 20 and humidity < 40:  # Example for drought
        drought_chance = 1
    if rainfall > 100 and humidity < 50:  # Example for floods
        flood_chance = 1
    if temp > 30 and humidity < 30:  # Example for bushfires
        bushfire_chance = 1
    
    return pd.Series([drought_chance, flood_chance, bushfire_chance])

# Apply the disaster calculation
disaster_chances = grouped.apply(calculate_disasters, axis=1)
disaster_chances.columns = ['Drought', 'Flood', 'Bushfire']
grouped = pd.concat([grouped, disaster_chances], axis=1)

# Set multi-index for better display
grouped.set_index(['OfficialNameSuburb', 'Month'], inplace=True)

# Display the results
print(grouped)
