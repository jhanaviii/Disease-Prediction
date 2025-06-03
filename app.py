from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    with open("random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()

# Load and clean the dataset to extract valid symptoms
try:
    train = pd.read_csv("/Applications/general/Projects/Disease-Prediction-using-ML-main/Training.csv")
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]  # Remove 'Unnamed' columns
    features = train.drop('prognosis', axis=1).columns.tolist()  # Extract symptom names
except FileNotFoundError:
    print("Error: Training dataset not found.")
    exit()

# Function to fuzzy match user input
def fuzzy_match_input(partial_input, valid_symptoms, threshold=60):
    return [match[0] for match in process.extractBests(partial_input, valid_symptoms, score_cutoff=threshold)]

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for dynamic suggestions
@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '').lower().strip()
    if not query:
        return jsonify([])
    suggestions = fuzzy_match_input(query, features)
    return jsonify(suggestions)

# Endpoint for disease prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("symptoms", [])
        if not data:
            return jsonify({"error": "No symptoms provided."}), 400

        # Create input vector: 1 if symptom is present, else 0
        input_data = pd.DataFrame([[1 if feature in data else 0 for feature in features]], columns=features)

        # Predict disease
        prediction = model.predict(input_data)[0]
        return jsonify({"predicted_disease": prediction})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
