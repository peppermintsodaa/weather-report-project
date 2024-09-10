#!/usr/bin/env python

import pandas as pd
import numpy as np

#read excel file
ica_data = pd.read_excel('datasets/ICA-Historical-Catastrophe-List-April-2022.xlsx', skiprows=7, header=0)

print(ica_data.info())
print(ica_data.head())

#make everything lower case for consistency
ica_data.columns = [col.strip().lower() for col in ica_data.columns]
ica_data[ica_data.select_dtypes(['object']).columns] = ica_data.select_dtypes(['object']).apply(lambda x: x.str.lower())


#remove rows that are missing critical information
ica_data.dropna(subset='event name', inplace=True)
ica_data.dropna(subset='cat event start', inplace=True)
ica_data.dropna(subset='type',inplace=True)

#convvert dates to datetime format
ica_data['cat event start'] = pd.to_datetime(ica_data['cat event start'], errors='coerce')
ica_data['cat event finish'] = pd.to_datetime(ica_data['cat event finish'], errors='coerce')


#drop columns with too many missing values or irrelevant data
dropped_columns = ['region / town', 'postcodes', 'download link', 'ica metadata record']
ica_data.drop(columns=dropped_columns, inplace=True)


#standardise type and state and strip whitespace
ica_data['type'] = ica_data['type'].str.strip().str.lower()
ica_data['state'] = ica_data['state'].str.strip().str.upper()


#i think it is helpful to calculate the median for that year in these columns rather than remove or leave them NaN!!!!!

#get the year
ica_data['year'] = ica_data['cat event start'].dt.year

claims_columns = [
    'total claims received', 
    'domestic building claims', 
    'domestic content claims', 
    'domestic motor claims', 
    'domestic other claims', 
    'commercial property claims', 
    'commercial motor', 
    'commercial bi claims', 
    'commercial other claims', 
    'commercial crop claims'
]

for col in claims_columns:
    if ica_data[col].dtype == 'float64':
        #calculate the median by year
        median_values = ica_data.groupby('year')[col].median()
        
        #fill missing values with the median of the corresponding year
        for year in median_values.index:
            ica_data.loc[(ica_data['year'] == year) & (ica_data[col].isnull()), col] = median_values[year]
    else:
        #fill categorical values
        ica_data[col].fillna('Unavailable', inplace=True)



#remove missing entries from cat event finish
ica_data.dropna(subset=['cat event finish'], inplace=True)

print(ica_data.info())
print(ica_data.head())

