# import libraries
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import model_1_insurance_advanced as insurance_model
import model_weather_new as weather_model

app = Flask(__name__)
cors = CORS(app)

#### ROUTING ####
@app.route('/')
def index():
    return render_template('index.html')

#### PREDICTING INSURANCE CLAIMS ####
@app.route('/get_claim', methods=['GET'])
@cross_origin()
def get_claim():
    state = request.args.get('state', type=str)
    disaster_type = request.args.get('disaster_type', type=str)
    method = request.args.get('method', type=str)
    new_data = insurance_model.generate_new_data(state, disaster_type)
    
    if method == 'model':
        # Use machine learning model to predict insurance claim amount
        loss, claim_amount = (insurance_model.predict_cost_with_model(new_data),
                              insurance_model.predict_amount_with_model(new_data)) 
        claim_individual = round(loss / (claim_amount+1), 2)
    else:
        # Use rule-based approach to calculate insurance claim amount
        claim_individual = insurance_model.calculate_insurance_amount(disaster_type, state)
    
    return jsonify({'claim_amount': claim_individual})

@app.route('/common_disasters', methods=['GET'])
def common_disasters():
    common = insurance_model.data.groupby('state')['type'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    return jsonify(common)

#### PREDICTING WEATHER ####
@app.route('/get_suburbs')
@cross_origin()
def get_suburbs():
    suburbs = weather_model.suburb_clustered.copy()
    suburbs = suburbs['OfficialNameSuburb'].sort_values()
    return jsonify({'suburbs': list(suburbs)})

@app.route('/predict_weather')
@cross_origin()
def predict_weather():
    month = request.args.get('month', type=str)
    suburb = request.args.get('suburb', type=str)

    grouped = weather_model.grouped.copy()
    matching_data = grouped[(grouped['Month'] == month) & (grouped['OfficialNameSuburb'] == suburb)]

    event_chances = weather_model.calculate_disaster(matching_data.to_numpy()[0])

    return jsonify({'chances': event_chances.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
