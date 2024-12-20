import joblib, numpy as np
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model('src/model.keras')


# Load the scaler
scaler = joblib.load('src/scaler.pkl')


# Prediction based on user input
while True:
    rainfall    = float(input('Rainfall (mm) per month: '))
    humidity    = float(input('Humidity (%)           : '))
    temperature = float(input('Temperature (°C)       : '))

    new_data = np.array([[rainfall, humidity, temperature]])

    new_data_scaled = scaler.transform(new_data)
    predicted_producion_per_harvest_area = model.predict(new_data_scaled)

    print(f'Predicted production per harvest area (ton/ha):\n > { predicted_producion_per_harvest_area[0][0] }')
