import pandas as pd 

# load datasets
ica_data = pd.read_csv('cosc2669-or-cosc2186-WIL-project/datasets_cleaned/cleaned_ica_data.csv')
disaster_data = pd.read_csv('cosc2669-or-cosc2186-WIL-project/datasets_cleaned/disaster-2023-cleaned.csv')

#i just removed one of the states column


columns_to_remove = ['NSW', 'VIC', 'QLD', 'ACT', 'TAS', 'SA', 'NT', 'WA']
ica_data_cleaned = ica_data.drop(columns=columns_to_remove)

disaster_data.columns = disaster_data.columns.str.lower()



#i merged the datasets on event name and event start

# merge on event start
merged_start = pd.merge(
    ica_data_cleaned,
    disaster_data,
    left_on='event start',
    right_on='start date',
    how='inner'
)

# merge on event name
merged_name = pd.merge(
    ica_data_cleaned,
    disaster_data,
    left_on='event name',
    right_on='event',
    how='inner'
)

#concat them together
merged_data = pd.concat([merged_start, merged_name]).drop_duplicates()

print(merged_data.info())

merged_data.to_csv('cosc2669-or-cosc2186-WIL-project/datasets_cleaned/merged_dataset.csv', index=False)


