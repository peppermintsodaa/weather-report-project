from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'

# # Define the base amounts for each disaster type
# disaster_base_amounts = {
#     'flooding': 1000000,
#     'hailstorm': 500000,
#     'bushfire': 750000
# }

# # Define multipliers for each state
# state_multipliers = {
#     'NSW': 1.2,
#     'VIC': 1.1,
#     'SA': 1.05,
#     'QLD': 1.15,
#     'WA': 1.0,  # Assuming default multiplier if not specified
#     'TAS': 1.0,
#     'NT': 1.0,
#     'ACT': 1.0
# }

# def calculate_insurance_amount(disaster_type, state):
#     base_amount = disaster_base_amounts.get(disaster_type.lower(), 0)
#     multiplier = state_multipliers.get(state.upper(), 1)
#     insurance_amount = base_amount * multiplier
#     return insurance_amount

# # Load and prepare the dataset
# data = pd.read_csv('datasets_cleaned/merged_dataset.csv')

# # Select features and target
# target = data['original loss value']

# # Convert categorical features to numerical
# features = pd.get_dummies(data['type'])

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Initialize and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Function to predict with the model
# def predict_with_model(state, disaster_type):
#     # Create a DataFrame with the disaster type
#     new_data = pd.DataFrame({
#         'type_' + disaster_type.lower(): [1]
#     })
    
#     # Ensure all columns are present
#     for col in X_train.columns:
#         if col not in new_data.columns:
#             new_data[col] = 0
#     new_data = new_data[X_train.columns]
    
#     # Predict insurance claim amount
#     insurance_claim_prediction = model.predict(new_data)
#     return insurance_claim_prediction[0]

# Define the base amounts for each disaster type
disaster_base_amounts = {
    'flooding': 1000000,
    'storm': 500000,
    'bushfire': 750000
}

# Define multipliers for each state
state_multipliers = {
    'NSW': 1.2,
    'VIC': 1.1,
    'SA': 1.05,
    'QLD': 1.15
}

def calculate_insurance_amount(disaster_type, state):
    # Get the base amount for the given disaster type
    base_amount = disaster_base_amounts.get(disaster_type.lower(), 0)
    
    # Get the multiplier for the given state
    multiplier = state_multipliers.get(state.upper(), 1)
    
    # Calculate the insurance amount
    insurance_amount = base_amount * multiplier
    
    return insurance_amount

# Load the dataset
data = pd.read_csv('datasets_cleaned/merged_dataset.csv')
#data = pd.read_csv('datasets_cleaned/ica-2024-cleaned.csv')

# Select features and target
target_1 = data['original loss value']
target_2 = data['claims count']

# Convert categorical features to numerical
#features = pd.get_dummies(data[['state','type']])
#features = pd.concat([features, data[['normalised loss value (2022)', 'claims count']]], axis = 1)
# this is if you're using cleaned_ica_data.csv
features = pd.get_dummies(data['type'])
features = pd.concat([features, data[['nsw', 'vic', 'qld', 'act', 'tas', 'sa', 'nt', 'wa', 'waters']]], axis = 1)

# Train-Test Split
X1_train, X1_test, y1_train, y1_test = train_test_split(features, target_1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(features, target_2, test_size=0.2, random_state=42)

# Initialize and train the model
model_1 = RandomForestRegressor(max_depth = 3, random_state = 42)
#model_1 = LinearRegression()
model_1.fit(X1_train, y1_train)

# Make predictions
y1_pred = model_1.predict(X1_test)

# Evaluate the model
mae = mean_absolute_error(y1_test, y1_pred)
mse = mean_squared_error(y1_test, y1_pred)

print(f"R2 Score 1: {model_1.score(X1_test, y1_test)}")
print(f"Mean Absolute Error 1: {mae}")
print(f"Mean Squared Error 1: {mse}")
print()

model_2 = DecisionTreeRegressor(random_state = 42)
model_2.fit(X2_train, y2_train)

# Make predictions
y2_pred = model_2.predict(X2_test)

# Evaluate the model
mae = mean_absolute_error(y2_test, y2_pred)
mse = mean_squared_error(y2_test, y2_pred)

print(f"R2 Score 1: {model_2.score(X2_test, y2_test)}")
print(f"Mean Absolute Error 1: {mae}")
print(f"Mean Squared Error 1: {mse}")
print()

def generate_new_data(state, disaster_type):
    new_data = pd.DataFrame({
        state: 1,
        disaster_type: 1
    }, index = range(1))

    new_data = new_data.reindex(columns=X1_train.columns, fill_value=0)
    return new_data

# Predict New Data Using Machine Learning Model
def predict_cost_with_model(new_data):
    # Predict insurance claim amount
    insurance_claim_prediction = model_1.predict(new_data)
    return insurance_claim_prediction[0]

def predict_amount_with_model(new_data):
    claim_amount = model_2.predict(new_data)
    return claim_amount[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_claim', methods=['GET'])
@cross_origin()
def get_claim():
    state = request.args.get('state', type=str)
    disaster_type = request.args.get('disaster_type', type=str)
    method = request.args.get('method', type=str)
    new_data = generate_new_data(state, disaster_type)
    
    if method == 'model':
        # Use machine learning model to predict insurance claim amount
        loss, claim_amount = (predict_cost_with_model(new_data), predict_amount_with_model(new_data)) 
        claim_individual = round(loss / claim_amount, 2)
    else:
        # Use rule-based approach to calculate insurance claim amount
        claim_individual = calculate_insurance_amount(disaster_type, state)
    
    return jsonify({'claim_amount': claim_individual})

@app.route('/common_disasters', methods=['GET'])
def common_disasters():
    common = data.groupby('state')['type'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    return jsonify(common)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
