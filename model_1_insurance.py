import pandas as pd

# Define the base amounts for each disaster type
disaster_base_amounts = {
    'flooding': 1000000,
    'hailstorm': 500000,
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('datasets_cleaned/merged_dataset.csv')

# Select features and target
target = data['original loss value']

# Convert categorical features to numerical
features = pd.get_dummies(data['type'])
#features = pd.concat([features, data[['nsw', 'vic', 'qld', 'act', 'tas', 'sa', 'nt', 'wa', 'waters']]], axis = 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 Score: {model.score(X_test, y_test)}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Predict New Data Using Machine Learning Model
def predict_with_model(state, disaster_type):
    new_data = pd.DataFrame({
        'state': [state],
        'type': [disaster_type]
    })

    # Convert new data to match the format of the training data
    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

    # Predict insurance claim amount
    insurance_claim_prediction = model.predict(new_data)
    return insurance_claim_prediction[0]

def predict_insurance_claim(state, disaster_type, use_model=True):
    if use_model:
        # Use machine learning model to predict insurance claim amount
        return predict_with_model(state, disaster_type)
    else:
        # Use rule-based approach to calculate insurance claim amount
        return calculate_insurance_amount(disaster_type, state)

# Example usage
state = 'NSW'
disaster_type = 'bushfire'

# Decide which method to use
insurance_claim = predict_insurance_claim(state, disaster_type, use_model=True)
print(f"Predicted Insurance Claim Amount: ${insurance_claim:,.2f}")

insurance_claim = predict_insurance_claim(state, disaster_type, use_model=False)
print(f"Insurance Claim Amount (Rule-Based): ${insurance_claim:,.2f}")

# Find the most commonly occurred disasters for each state
#common_disasters = data.groupby(['NSW', 'VIC', 'QLD', 'ACT', 'TAS', 'SA', 'NT', 'WA'])['type'].agg(lambda x: x.value_counts().idxmax())
common_disasters = data.groupby('state')['type'].agg(lambda x: x.value_counts().idxmax())

print("Most Common Disasters for Each State:")
#print(common_disasters)
