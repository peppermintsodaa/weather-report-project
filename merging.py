import pandas as pd
import numpy as np
from sklearn import linear_model

# load datasets
ica_data = pd.read_csv('datasets_cleaned/cleaned_ica_data.csv')
disaster_data = pd.read_csv('datasets_cleaned/disaster-2023-cleaned.csv')

disaster_data.columns = disaster_data.columns.str.lower()
ica_data.columns = ica_data.columns.str.lower()

# rename columns in disaster_data to match ica_data
disaster_data.rename(columns = {'event':'event name',
                                'zone':'state',
                                'category':'type',
                                'start date':'event start',
                                'end date':'event finish',
                                'region':'location',
                                'insured cost':'original loss value'}, inplace = True)

# make all types lower
disaster_data['type'] = disaster_data['type'].str.lower()

#concat them together
merged_data = pd.concat([ica_data, disaster_data]).drop_duplicates()

# keep categories consistent
merged_data['type'] = merged_data['type'].replace({'flooding':'flood', 'hailstorm':'storm'})

#normalise missing entries for normalised loss value (2022) as in 2024_ica_cleaner.py
normalization_factor = 1.05
merged_data['normalised loss value (2022)'] = \
    merged_data['normalised loss value (2022)'].fillna(merged_data['original loss value'] * normalization_factor)

# drop not required columns
merged_data.drop(columns = ['cat name', 'description'], inplace = True)

# impute all values with missing values in "waters"
merged_data['waters'] = merged_data['waters'].fillna(0).astype('int64')

# predict claims count
# the following code is adapted from the COSC2820 Week 5 Lab (Sarwar 2024)
## all rows with missing count value
missing_count = merged_data.loc[merged_data.isnull()['claims count'], 'claims count']
## all rows with a count value
no_missing_count = merged_data.dropna(subset='claims count')
## all unique costs
count_ls = no_missing_count['claims count'].unique().tolist()

## imputing rest of missing values through linear regression
types = pd.get_dummies(no_missing_count['type'])
merged_data = pd.concat([merged_data, pd.get_dummies(merged_data['type'])], axis = 1)

x_train = pd.concat([types, no_missing_count.iloc[:, 7:9], no_missing_count.iloc[:, 10:]], axis = 1)
y_train = no_missing_count['claims count']
clf = linear_model.LinearRegression()

clf.fit(x_train.values, y_train)

## function to predict costs
def predict_count(x):
    x_predict = clf.predict([x])
    return min(count_ls, key=lambda y:abs(y-x_predict))

## impute all missing costs
cols = list(x_train)
merged_data.loc[missing_count.index, 'claims count'] = merged_data.loc[missing_count.index].apply(lambda row: predict_count(row[cols]), axis=1)
# END OF ADAPTED CODE

# drop not required columns
merged_data.drop(columns = list(merged_data)[19:], inplace = True)

# convert start and end dates to period type
merged_data['event start'] = merged_data['event start'].apply(lambda x: pd.Period(x, freq='D'))
merged_data['event finish'] = merged_data['event finish'].apply(lambda x: pd.Period(x, freq='D'))

merged_data = merged_data.reset_index(drop=True)

# check for duplicates
state_ls = cols[10:]
condition_1 = merged_data.duplicated(['event start', 'event finish', 'year'] + state_ls, keep=False)

duplicates = merged_data[condition_1].sort_values(by=['event start', 'event finish', 'year'] + state_ls, ascending=False)

# subset by rows from ica dataset
condition_2 = duplicates['state'].str.len() <= 3
condition_3 = duplicates['type'].isin(['earthquake', 'storm', 'flood', 'cyclone', 'tornado'])

duplicates_first_only = duplicates[condition_2 & condition_3]
duplicates_first_only = duplicates_first_only.sort_values(by=['event start', 'event finish', 'type', 'year'], ascending=False).reset_index()

# subset by rows from disaster dataset
duplicates_second_only = duplicates[~condition_2 & condition_3]
duplicates_second_only = duplicates_second_only.sort_values(by=['event start', 'event finish', 'type', 'year'], ascending=False)\
                            .drop(index=[573, 635]).reset_index()

# replace values from ica dataset onto disaster dataset since we are keeping rows from second dataset only
duplicates_second_only['normalised loss value (2022)'] = duplicates_first_only['normalised loss value (2022)'].values
duplicates_second_only['claims count'] = duplicates_first_only['claims count'].values

# reset index and drop index column
duplicates_first_only.index = duplicates_first_only['index']
duplicates_second_only.index = duplicates_second_only['index']

duplicates_first_only = duplicates_first_only.drop(columns = 'index')
duplicates_second_only = duplicates_second_only.drop(columns = 'index')

# apply values onto merged dataset and drop unnecessary rows and state column
cols = ['original loss value', 'normalised loss value (2022)', 'claims count']

merged_data.loc[merged_data.index.isin(duplicates_second_only.index), cols] = \
    duplicates_second_only.loc[duplicates_second_only.index.isin(merged_data.index), cols].values
merged_data = merged_data.drop(columns = 'state', index = duplicates_first_only.index)

# explode all locations onto separate rows
merged_data['location'] = merged_data['location'].apply(lambda x: [r.strip() for r in x.split(',')])
merged_data = merged_data.explode('location')

# sort by descending start date
merged_data = merged_data.sort_values('event start', ascending = False)

merged_data.to_csv('datasets_cleaned/merged_dataset.csv', index=False)


