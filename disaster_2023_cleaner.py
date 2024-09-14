import pandas as pd
import re
import difflib
import matplotlib.pyplot as plt
import numpy as np

# read excel file
disaster_2023 = pd.read_excel('datasets/disaster-mapper-data-21-03-2023.xlsx')

# eliminate all header rows
disaster_2023.dropna(subset=['Zone'], inplace = True)

# strip whitespace off all columns
disaster_2023.rename({'Event ':'Event'}, axis = 1, inplace = True)

# drop all unnecessary columns
disaster_2023.drop(columns = ['Fatalities', 'Injured', 'URL', 'Description', 'Source(s)'], inplace = True)

# fix capitalisation errors in "Zone", "Region", and "Category"
disaster_2023['Zone'].replace({"New South wales":"New South Wales",
                               "Western australia":"Western Australia",
                               "victoria":"Victoria",
                               "New South  Wales":"New South Wales"}, inplace = True)
disaster_2023['Region'].replace({"sydney":"Sydney",
                                 "North sydney":"North Sydney",
                                 "bundaberg":"Bundaberg"}, inplace = True)
disaster_2023['Category'].replace({"flood":"Flood"}, inplace = True)

# function to convert to standard date format
def convert_to_std_date(x):
    dmy = x.split('/')
    day = dmy[0]
    month = dmy[1]
    year = dmy[2]

    new_time = '{}-{}-{} 00:00:00'.format(year, month, day)
    return new_time

# condition to find all non-null values with irregular date format
condition = ~disaster_2023['Start Date'].isnull() & disaster_2023['Start Date'].str.contains("/")
# fix inconsistent dates in "Start Date"
disaster_2023.loc[condition, 'Start Date'] = disaster_2023.loc[condition, 'Start Date'].apply(lambda x: convert_to_std_date(x))

# replace "`" with correct end date according to URL provided ()"incident lasted 10 days")
disaster_2023['End Date'] = disaster_2023['End Date'].replace({"`":'1944-08-15 00:00:00'})
# condition to find all non-null values with irregular date format
condition = ~disaster_2023['End Date'].isnull() & disaster_2023['End Date'].str.contains("/")
# fix inconsistent dates in "End Date" using condition above
disaster_2023.loc[condition, 'End Date'] = disaster_2023.loc[condition, 'End Date'].apply(lambda x: convert_to_std_date(x))

# replace all "Not Available" values in "Insured Cost" to null value
disaster_2023['Insured Cost'] = disaster_2023['Insured Cost'].replace({"Not Available":None})
# fix inconsistent costs in "Insured Cost", converting any GBP cost to AUD
disaster_2023['Insured Cost'] = disaster_2023['Insured Cost'].replace({"response effort exceeded $60 Mil":"60000000",
                                                                       "£1,000,000":"2000000",
                                                                       "£1,500,000":"3000000",
                                                                       "£300,000-400,000":"700000",
                                                                       "$31 ,000,000":"31000000",
                                                                       "$17,81,599,484":"1781599484"})

# convert "Start Date" and "End Date" to period data type as dates older than 1860 exist
disaster_2023['Start Date'] = disaster_2023['Start Date'].apply(lambda x: pd.Period(x, freq='D'))
disaster_2023['End Date'] = disaster_2023['End Date'].apply(lambda x: pd.Period(x, freq='D'))
# convert insured cost to float values
disaster_2023['Insured Cost'] = pd.to_numeric(disaster_2023['Insured Cost'], errors = 'coerce')

# vectorise regions
disaster_2023.loc[:, 'NSW'] = 0
disaster_2023.loc[:, 'VIC'] = 0
disaster_2023.loc[:, 'QLD'] = 0
disaster_2023.loc[:, 'ACT'] = 0
disaster_2023.loc[:, 'TAS'] = 0
disaster_2023.loc[:, 'SA'] = 0
disaster_2023.loc[:, 'NT'] = 0
disaster_2023.loc[:, 'WA'] = 0
disaster_2023.loc[:, 'Waters'] = 0

# count all regions present in "Zone" column into respective region columns
disaster_2023.loc[disaster_2023['Zone'].str.contains("New South Wales"), 'NSW'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Victoria"), 'VIC'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Queensland"), 'QLD'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Australian Capital Territory"), 'ACT'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Tasmania"), 'TAS'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("South Australia"), 'SA'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Northern Territory"), 'NT'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Western Australia"), 'WA'] = 1
# for national and offshore events
disaster_2023.loc[disaster_2023['Zone'].str.contains("National"), 'NSW':'WA'] = 1
disaster_2023.loc[disaster_2023['Zone'].str.contains("Offshore"), 'Waters'] = 1

# impute all missing insured costs
