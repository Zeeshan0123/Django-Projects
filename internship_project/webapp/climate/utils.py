import joblib
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_PATH = "model.pkl"

def train_model(csv_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path)
    data = data.set_index('Year')

    # Extract features and target
    X = data[['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']]
    y = data['LandAndOceanAverageTemperature']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

    # Train the model
    knn_model = KNeighborsRegressor(n_neighbors = 8)  # we have find the best k value in model testing phase  
    knn_model.fit(X_train, y_train)

   

    # Save the model
    joblib.dump(knn_model, MODEL_PATH)


def predict_temperature(features):
    # Load the model
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet. Please train the model first."}

    knn_model = joblib.load(MODEL_PATH)
    
    # Predict using the model
    prediction = knn_model.predict([features])
    return prediction[0]