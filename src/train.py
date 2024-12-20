import os, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Set-up working directory
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def resolve_path(filename):
    return os.path.join(WORKING_DIRECTORY, filename)


# Load dataset
data = pd.read_csv(resolve_path('dataset.csv'))


# Feature engineering
data['Produksi_per_Luas_Panen'] = data['Produksi'] / data['Luas Panen']


# Select the features and target variable
X = data[['Curah hujan', 'Kelembapan', 'Suhu rata-rata']].values
y = data['Produksi_per_Luas_Panen'].values


# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Build the neural network model
model = Sequential()


# Neural Network Layers
model.add(Dense(128, input_dim=3, activation='relu'))   # Hidden layer
model.add(Dense(96, activation='relu'))                 # Hidden layer
model.add(Dense(64, activation='relu'))                 # Hidden layer
model.add(Dense(1))                                     # Output layer


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=64, restore_best_weights=True)
model.fit(X_train, y_train, epochs=256, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


# Evaluate the model
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

with open(resolve_path('log.txt'), 'w') as f:
    f.write(f'R Squared          : {r2}\n')
    f.write(f'Mean Absolute Error: {mae}\n')
    f.write(f'Mean Squared Error : {mse}\n')


# Save the trained model
model.save(resolve_path('model.keras'))


# Save the scaler
joblib.dump(scaler, resolve_path('scaler.pkl'))