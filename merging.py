import pandas as pd 

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

#concat them together
merged_data = pd.concat([ica_data, disaster_data]).drop_duplicates()

#normalise missing entries for normalised loss value (2022) as in 2024_ica_cleaner.py
normalization_factor = 1.05
merged_data['normalised loss value (2022)'].fillna(
    merged_data['original loss value'] * normalization_factor, inplace=True
)

# drop not required columns
merged_data.drop(columns = ['cat name', 'description'], inplace = True)

print(merged_data.info())

merged_data.to_csv('datasets_cleaned/merged_dataset.csv', index=False)


