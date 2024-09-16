import pandas as pd
from sklearn import linear_model

# read excel file
disaster_2023 = pd.read_excel('datasets/disaster-mapper-data-21-03-2023.xlsx')

# eliminate all header rows
disaster_2023.dropna(subset=['Zone'], inplace = True)

# strip whitespace off all columns
disaster_2023.rename({'Event ':'Event'}, axis = 1, inplace = True)

# drop all unnecessary columns
disaster_2023.drop(columns = ['Fatalities', 'Injured', 'URL', 'Description', 'Source(s)'], inplace = True)

# fix capitalisation errors in "Zone", "Region", and "Category"
disaster_2023['Zone'] = disaster_2023['Zone'].replace({"New South wales":"New South Wales",
                                                       "Western australia":"Western Australia",
                                                       "victoria":"Victoria",
                                                       "New South  Wales":"New South Wales"})
disaster_2023['Region'] = disaster_2023['Region'].replace({"sydney":"Sydney",
                                                           "North sydney":"North Sydney",
                                                           "bundaberg":"Bundaberg"})
disaster_2023['Category'] = disaster_2023['Category'].replace({"flood":"Flood"})

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

# replace "`" with correct end date according to URL provided ("incident lasted 10 days")
disaster_2023['End Date'] = disaster_2023['End Date'].replace({"`":'1944-08-15 00:00:00'})
# condition to find all non-null values with irregular date format
condition = ~disaster_2023['End Date'].isnull() & disaster_2023['End Date'].str.contains("/")
# fix inconsistent dates in "End Date" using condition above
disaster_2023.loc[condition, 'End Date'] = disaster_2023.loc[condition, 'End Date'].apply(lambda x: convert_to_std_date(x))
# imputing rest of the ending dates using the URLs provided
disaster_2023.loc[20, 'End Date'] = '2006-02-07 00:00:00'
disaster_2023.loc[39, 'End Date'] = '2007-12-20 00:00:00'
disaster_2023.loc[102, 'End Date'] = '2019-03-22 00:00:00'
disaster_2023.loc[103, 'End Date'] = '2018-03-25 00:00:00'
disaster_2023.loc[106, 'End Date'] = '2013-01-31 00:00:00'
disaster_2023.loc[203, 'End Date'] = '1893-03-09 00:00:00'
disaster_2023.loc[231, 'End Date'] = '2005-11-09 00:00:00'
disaster_2023.loc[280, 'End Date'] = '2022-04-07 00:00:00'
disaster_2023.loc[357, 'End Date'] = '2015-12-17 00:00:00'
disaster_2023.loc[431, 'End Date'] = '1858-08-06 00:00:00'
# following events might have happened for only one day
condition = disaster_2023['End Date'].isnull()
disaster_2023.loc[condition, 'End Date'] = disaster_2023.loc[condition, 'Start Date']

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

# exploding rows by city / town
disaster_2023['Region'] = disaster_2023['Region'].apply(lambda x: [r.strip() for r in x.split(',')])
disaster_2023 = disaster_2023.explode('Region')

# create new "Year" column
disaster_2023['Year'] = disaster_2023['Start Date'].apply(lambda x : str(x).split('-')[0]).astype('int64')
# imputing all insured costs with known median for year
disaster_2023['Insured Cost'] = disaster_2023['Insured Cost'].fillna(disaster_2023.groupby('Year')['Insured Cost'].transform('median'))

# the following code is adapted from 
# all rows with missing cost value
missing_cost = disaster_2023.loc[disaster_2023.isnull()['Insured Cost'], 'Insured Cost']
# all rows with a cost value
no_missing_cost = disaster_2023.dropna(subset='Insured Cost')
# all unique costs
cost_ls = no_missing_cost['Insured Cost'].unique().tolist()

# imputing rest of missing values through linear regression
# taken from COSC2820 week 5 practical
cols = list(disaster_2023)

cols = cols[7:]
x_train = no_missing_cost[cols]
y_train = no_missing_cost['Insured Cost']
regr = linear_model.LinearRegression()

regr.fit(x_train.values, y_train)

# function to predict costs
def predict_cost(x):
    x_predict = regr.predict([x])
    return min(cost_ls, key=lambda y:abs(y-x_predict))

# impute all missing costs
disaster_2023.loc[missing_cost.index, 'Insured Cost'] = disaster_2023.loc[missing_cost.index].apply(lambda row: predict_cost(row[cols]), axis=1)

# rearrange columns
cols = list(disaster_2023)
cols = cols[:6] + [cols[-1]] + cols[6:-1]
disaster_2023 = disaster_2023[cols]

# printing final data
print(disaster_2023.info())
print(disaster_2023.head())

disaster_2023.to_csv('datasets_cleaned/disaster-2023-cleaned.csv', index=False)
