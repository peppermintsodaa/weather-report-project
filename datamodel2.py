import pandas as pd

# Load your weather dataset
df = pd.read_csv('datasets_cleaned/cleaned_weather_data.csv')

# Clean and preprocess the data
df.fillna(method='ffill', inplace=True)  # Fill missing values if necessary

# Define state boundaries for grouping based on latitude and longitude
def assign_state(row):
    latitude = row['Latitude']
    longitude = row['Longitude']
    
    # Define the boundaries for each state
    if -40 <= latitude <= -34 and 140 <= longitude <= 150:  # South Australia
        return 'South Australia'
    elif -40 <= latitude <= -35 and 141 <= longitude <= 149:  # Victoria
        return 'Victoria'
    elif -28 <= latitude <= -10 and 138 <= longitude <= 154:  # Queensland
        return 'Queensland'
    elif -35 <= latitude <= -32 and 150 <= longitude <= 154:  # New South Wales
        return 'New South Wales'
    elif -43 <= latitude <= -39 and 146 <= longitude <= 148:  # Tasmania
        return 'Tasmania'
    elif -10 <= latitude <= -24 and 130 <= longitude <= 138:  # Northern Territory
        return 'Northern Territory'
    elif -35 <= latitude <= -29 and 115 <= longitude <= 130:  # Western Australia
        return 'Western Australia'
    elif -35 <= latitude <= -34 and 149 <= longitude <= 151:  # Australian Capital Territory
        return 'Australian Capital Territory'
    else:
        return 'Unknown'

# Assign state based on latitude and longitude
df['State'] = df.apply(assign_state, axis=1)

# Filter out unknown states
df = df[df['State'] != 'Unknown']

# Group data by state and month
grouped = df.groupby(['State', 'Month']).agg({
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
grouped.set_index(['State', 'Month'], inplace=True)

# Display the results
print(grouped)
