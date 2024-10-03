import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Select features and target
target_1 = data['original loss value']
target_2 = data['claims count']

# Convert categorical features to numerical
features = pd.get_dummies(data['type'])
features = pd.concat([features, data[['nsw', 'vic', 'qld', 'act', 'tas', 'sa', 'nt', 'wa', 'waters']]], axis = 1)

# Train-Test Split
X1_train, X1_test, y1_train, y1_test = train_test_split(features, target_1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(features, target_2, test_size=0.2, random_state=42)

# Initialize and train the model
model_1 = RandomForestRegressor(max_depth = 3, random_state = 42)
model_1.fit(X1_train, y1_train)

model_2 = RandomForestRegressor(max_depth = 5, random_state = 42)
#model_2 = RandomForestRegressor(random_state = 42)
model_2.fit(X2_train, y2_train)

def evaluate(X1_train, y1_train, X2_train, y2_train):
    # Make predictions
    y1_pred = model_1.predict(X1_test)

    # Evaluate the model
    mae = mean_absolute_error(y1_test, y1_pred)
    mse = mean_squared_error(y1_test, y1_pred)

    print(f"R2 Score 1: {model_1.score(X1_test, y1_test)}")
    print(f"Mean Absolute Error 1: {mae}")
    print(f"Mean Squared Error 1: {mse}")
    print()

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

def predict_insurance_claim(state, disaster_type, new_data, use_model=True):
    if use_model:
        # Use machine learning model to predict insurance claim amount
        return (predict_cost_with_model(new_data), predict_amount_with_model(new_data)) 
    else:
        # Use rule-based approach to calculate insurance claim amount
        return calculate_insurance_amount(disaster_type, state)

# Example usage
# state = 'nsw'
# disaster_type = 'bushfire'

# new_data = generate_new_data(state, disaster_type)

# # # Decide which method to use
# loss, claim_amount = predict_insurance_claim(state, disaster_type, new_data, use_model=True)
# print(f"Predicted Insurance Claim Amount: ${loss:,.2f}")
# print(f"Predicted Insurance Claim Amount: {claim_amount:,.0f}")
# print(f"Predicted Insurance Claim Amount: ${loss / (claim_amount+1):,.0f}")

# insurance_claim = predict_insurance_claim(state, disaster_type, new_data, use_model=False)
# print(f"Insurance Claim Amount (Rule-Based): ${insurance_claim:,.0f}")
