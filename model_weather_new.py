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
weather_data.ffill(inplace=True)

# Merge weather_data with clusters to get latitude and longitude
weather_data = weather_data.merge(clusters[['ClusterID', 'Latitude', 'Longitude']], on='ClusterID')

# Merge with suburb_clustered to get suburb information
weather_data = weather_data.merge(suburb_clustered[['ClusterID', 'OfficialNameSuburb', 'OfficialNameState']], on='ClusterID')

weather_data_copy = weather_data.copy()

weather_data_copy['Month'] = weather_data_copy['Month'].replace({1: 'January',
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
                                                                12: 'December', })

# Group data by suburb and month
grouped = weather_data_copy.groupby(['OfficialNameSuburb', 'Month']).agg({
    'TemperatureMean': 'mean',          # Average temperature
    'RainSum': 'sum',                   # Total rainfall
    'RelativeHumidityMean': 'mean'      # Average humidity
}).reset_index()

# Calculate chances of disasters
# version of calculate_disasters() but using array
def calculate_disaster(row):
    temp = row[2]
    rainfall = row[3]
    humidity = row[4]

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

# Set multi-index for better display
#grouped.set_index(['OfficialNameSuburb', 'Month'], inplace=True)

# Display the results
#print(grouped)

#grouped.to_csv('datasets_cleaned/grouped_suburbs.csv', index=False)

#has_data = grouped[['Drought', 'Flood', 'Bushfire']].any()
#print(has_data)

#non_zero_counts = (grouped[['Drought', 'Flood', 'Bushfire']] != 0).sum()
#print(non_zero_counts)

#non_zero_rows = grouped[(grouped['Drought'] != 0) | (grouped['Flood'] != 0) | (grouped['Bushfire'] != 0)]
#print(non_zero_rows)

def generate_suburb_event_list(grouped):
    disaster_chances = grouped.apply(calculate_disasters, axis=1)
    disaster_chances.columns = ['Drought', 'Flood', 'Bushfire']
    grouped = pd.concat([grouped, disaster_chances], axis=1)
    non_zero_rows = grouped[(grouped['Drought'] != 0) | 
                            (grouped['Flood'] != 0) | 
                            (grouped['Bushfire'] != 0)]
    
    with open('suburbs_with_weather_events.txt', 'w') as f:
        for row in non_zero_rows.to_numpy():
            weather_events = []
            event_string = ""

            if row[5] == 1:
                weather_events.append('Drought')
            if row[6] == 1:
                weather_events.append('Flood')
            if row[7] == 1:
                weather_events.append('Bushfire')
            for event in weather_events:
                event_string += event + ','

            f.write('{}, {}: {}\n'.format(row[0], row[1], event_string[:-1]))

import numpy as np
from sklearn.metrics import classification_report

def evaluate(grouped):
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

# execute functions
generate_suburb_event_list(grouped)