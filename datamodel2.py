import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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



has_data = grouped[['Drought', 'Flood', 'Bushfire']].any()
print(has_data)

non_zero_counts = (grouped[['Drought', 'Flood', 'Bushfire']] != 0).sum()
print(non_zero_counts)

non_zero_rows = grouped[(grouped['Drought'] != 0) | (grouped['Flood'] != 0) | (grouped['Bushfire'] != 0)]
print(non_zero_rows)


import numpy as np
from sklearn.metrics import classification_report

# Generate synthetic true labels for testing purposes
np.random.seed(42)
grouped['TrueDrought'] = np.random.randint(0, 2, size=len(grouped))  # Synthetic true drought labels
grouped['TrueFlood'] = np.random.randint(0, 2, size=len(grouped))    # Synthetic true flood labels
grouped['TrueBushfire'] = np.random.randint(0, 2, size=len(grouped)) # Synthetic true bushfire labels

# --- Calculate metrics for Drought ---
y_true_drought = grouped['TrueDrought']  # Replace this with actual labels if available
y_pred_drought = grouped['Drought']      # Predicted drought events

# Display classification metrics for Drought
print("Drought Classification Report:")
print(classification_report(y_true_drought, y_pred_drought))

# --- Calculate metrics for Flood ---
y_true_flood = grouped['TrueFlood']  # Replace this with actual labels if available
y_pred_flood = grouped['Flood']      # Predicted flood events

# Display classification metrics for Flood
print("Flood Classification Report:")
print(classification_report(y_true_flood, y_pred_flood))

# --- Calculate metrics for Bushfire ---
y_true_bushfire = grouped['TrueBushfire']  # Replace this with actual labels if available
y_pred_bushfire = grouped['Bushfire']      # Predicted bushfire events

# Display classification metrics for Bushfire
print("Bushfire Classification Report:")
print(classification_report(y_true_bushfire, y_pred_bushfire))

# export to csv
grouped['Month'] = grouped['Month'].replace({1: 'January',
                                             2: 'February',
                                             3: 'March',
                                             4: 'April',
                                             5: 'May',
                                             6: 'June',
                                             7: 'July',
                                             8: 'August',
                                             9: 'September',
                                             10: 'October',
                                             11: 'November',
                                             12: 'December'})
grouped.drop(columns=['Drought', 'Flood', 'Bushfire']).to_csv('datasets_cleaned/grouped_suburbs.csv')

