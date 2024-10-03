import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# read merged weather dataset
#weather_data = pd.read_csv('datasets_cleaned/merged_weather_data.csv')
weather_data = pd.read_csv('datasets_cleaned/merged_weather_data_tiny.csv')

# let's predict the mean as a test
# get target and feature variables
#target_1 = ['ok', 'bushfire', 'flood', 'storm', 'cyclone']

feature_ls = list(weather_data)
feature_ls = ['Month', 'ClusterID']
#feature_ls = [feature_ls[0]] + feature_ls[4:6] + feature_ls[8:11]
features = weather_data[feature_ls]

# splitting train-test data
X1_train, X1_test, y1_train, y1_test = train_test_split(features, target_1, test_size=0.2, random_state=42)

# initialise model and train it
clf = LinearRegression()

clf.fit(X1_train, y1_train)

# predict
y1_pred = clf.predict(X1_test)

# evaluate model
mae = mean_absolute_error(y1_test, y1_pred)
mse = mean_squared_error(y1_test, y1_pred)

print(f"R2 Score 1: {clf.score(X1_test, y1_test)}")
print(f"Mean Absolute Error 1: {mae}")
print(f"Mean Squared Error 1: {mse}")
print()

# using model on new data
data_dict = {
    'ClusterID': 100412,
    #'TemperatureMax': 30,
    #'TemperatureMin': 17,
    'RainSum': 0,
    'RelativeHumidityMean': 72,
    'Year': 2024,
    'Month': 9,
    'Day': 17,
}

def predict_weather(data_dict):
    df = pd.DataFrame(data_dict, index = range(1))
    df = df.reindex(columns=X1_train.columns, fill_value=0)
    
    weather_prediction = clf.predict(df)
    return weather_prediction[0]

temp = predict_weather(data_dict)
print(f"Predicted Temperature: {temp:.2f}")