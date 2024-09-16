#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

#read excel file
ica_data = pd.read_excel('cosc2669-or-cosc2186-WIL-project/datasets/ICA-Historical-Normalised-Catastrophe-July-2024.xlsx', skiprows=9, header=0)



#always strip of whitespaces
ica_data.columns = [col.strip() for col in ica_data.columns]

#make everything lower case for consistency
ica_data.columns = [col.strip().lower() for col in ica_data.columns]

#convvert dates to datetime format
ica_data['event start'] = pd.to_datetime(ica_data['event start'], errors='coerce')
ica_data['event finish'] = pd.to_datetime(ica_data['event finish'], errors='coerce')

#remove rows that are missing critical information

ica_data.dropna(subset='event name', inplace=True)
ica_data.dropna(subset='event start', inplace=True)
ica_data.dropna(subset='type',inplace=True)
ica_data.dropna(subset='location',inplace=True)

#drop columns with too many missing values or irrelevant data
dropped_columns = ['domestic building claims', 
    'domestic content claims', 
    'domestic motor claims', 
    'domestic other claims', 
    'commercial property claims', 
    'commercial motor', 
    'commercial bi claims', 
    'commercial other claims', 
    'commercial crop claims',
    'town',
    'postcode',
    'financial year']

ica_data.drop(columns=dropped_columns, inplace=True)


#standardise type and state and strip whitespace
ica_data['type'] = ica_data['type'].str.strip().str.lower()
ica_data['state'] = ica_data['state'].str.strip().str.upper()

# handle multiple states
state_list = ['NSW', 'VIC', 'QLD', 'ACT', 'TAS', 'SA', 'NT', 'WA']  
for state in state_list:
    ica_data[state] = 0  

for index, row in ica_data.iterrows():
    if isinstance(row['state'], str):
        states = row['state'].split(',')
        for state in states:
            state = state.strip()
            if state in state_list:
                ica_data.at[index, state] = 1  



#i think it is helpful to calculate the median for that year in these columns rather than remove or leave them NaN!!!!!

#get the year
ica_data['year'] = ica_data['event start'].dt.year

claims_columns = [
    'claims count', 
    'original loss value', 
]


for col in claims_columns:
    if ica_data[col].dtype == 'float64':
        overall_median = ica_data[col].median()
        ica_data[col].fillna(overall_median, inplace=True)
    else:
        ica_data[col].fillna('Unavailable', inplace=True)

#normalise missing entries' for normalised loss value (2022)
normalization_factor = 1.05
ica_data['normalised loss value (2022)'].fillna(
    ica_data['original loss value'] * normalization_factor, inplace=True
)


print(ica_data.info())
print(ica_data.head())



ica_data.to_csv('cosc2669-or-cosc2186-WIL-project/datasets_cleaned/cleaned_ica_data.csv', index=False)
