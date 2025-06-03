import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fuzzywuzzy import process
import pickle

def load_and_clean_data():
    """Load and clean the training dataset."""
    train = pd.read_csv("/Applications/general/Projects/Disease-Prediction-using-ML-main/Training.csv")

    # Remove unnecessary columns like 'Unnamed: 133'
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]

    return train

def train_model(x_train, y_train):
    """Train Random Forest Classifier with fixed random state for reproducibility."""
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    return rf

def fuzzy_match_input(input_symptoms, valid_symptoms):
    """Match user input symptoms to valid symptoms using fuzzy matching."""
    matched_symptoms = []
    for symptom in input_symptoms:
        match, score = process.extractOne(symptom, valid_symptoms)
        if score > 60:  # Lowered threshold to 60 for better fuzzy matching
            print(f"Corrected '{symptom}' to '{match}'")
            matched_symptoms.append(match)
        else:
            print(f"Symptom '{symptom}' not recognized.")
    return matched_symptoms

def predict_user_symptoms(model, symptoms, features):
    """Predict the disease based on matched user symptoms."""
    # Generate input vector as a DataFrame with correct feature names
    input_data = pd.DataFrame([[1 if feature in symptoms else 0 for feature in features]],
                              columns=features)
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    print("Loading and cleaning data...")
    train = load_and_clean_data()

    # Prepare features (X) and labels (y)
    X = train.drop('prognosis', axis=1)
    y = train['prognosis']

    print("Preparing datasets...")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training model...")
    model = train_model(x_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy of Random Forest: {accuracy * 100:.2f}%")

    print("Saving the trained model...")
    with open("random_forest_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("\nInteractive Prediction")
    features = X.columns.tolist()
    print("Available Symptoms: \n" + ", ".join(features))

    while True:
        user_input = input("\nEnter symptoms separated by commas (or type 'exit' to quit): ").strip().lower()
        if user_input == 'exit':
            print("Exiting...")
            break

        # Process user input
        input_symptoms = [symptom.strip() for symptom in user_input.split(',')]
        matched_symptoms = fuzzy_match_input(input_symptoms, features)

        if not matched_symptoms:
            print("No valid symptoms found. Please try again.")
            continue

        print(f"Matched Symptoms: {', '.join(matched_symptoms)}")
        predicted_disease = predict_user_symptoms(model, matched_symptoms, features)
        print(f"Predicted Disease: {predicted_disease}")

if __name__ == "__main__":
    main()
