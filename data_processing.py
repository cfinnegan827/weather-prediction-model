import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stats import perc_error, abs_error, get_data

data = pd.read_csv('test.csv')


data['time'] = pd.to_datetime(data['time'])

data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['day'] = data['time'].dt.day


lags = 12
for feature in ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']:
    for lag in range(1, lags + 1):
        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)

# Drop rows with NaNs resulting from lagged features
data.dropna(inplace=True)

# Define features and target
features = [
    'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres', 'month', 'day',
    'tavg_lag_1', 'tavg_lag_2', 'tavg_lag_3', 'tavg_lag_4', 'tavg_lag_5', 'tavg_lag_6','tavg_lag_7', 'tavg_lag_8', 'tavg_lag_9', 'tavg_lag_10','tavg_lag_11', 'tavg_lag_12',
    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3','tmin_lag_4', 'tmin_lag_5', 'tmin_lag_6','tmin_lag_7', 'tmin_lag_8', 'tmin_lag_9', 'tmin_lag_10', 'tmin_lag_11', 'tmin_lag_12',
    'tmax_lag_1', 'tmax_lag_2', 'tmax_lag_3', 'tmax_lag_4', 'tmax_lag_5', 'tmax_lag_6','tmax_lag_7', 'tmax_lag_8', 'tmax_lag_9', 'tmax_lag_10', 'tmax_lag_11', 'tmax_lag_12',
    'prcp_lag_1', 'prcp_lag_2', 'prcp_lag_3', 'prcp_lag_4', 'prcp_lag_5', 'prcp_lag_6', 'prcp_lag_7', 'prcp_lag_8', 'prcp_lag_9', 'prcp_lag_10', 'prcp_lag_11', 'prcp_lag_12',
    'wdir_lag_1', 'wdir_lag_2', 'wdir_lag_3', 'wdir_lag_4', 'wdir_lag_5', 'wdir_lag_6', 'wdir_lag_7', 'wdir_lag_8', 'wdir_lag_9', 'wdir_lag_10', 'wdir_lag_11', 'wdir_lag_12',
    'wspd_lag_1', 'wspd_lag_2', 'wspd_lag_3', 'wspd_lag_4', 'wspd_lag_5', 'wspd_lag_6', 'wspd_lag_7', 'wspd_lag_8', 'wspd_lag_9', 'wspd_lag_10', 'wspd_lag_11', 'wspd_lag_12',
    'pres_lag_1', 'pres_lag_2', 'pres_lag_3', 'pres_lag_4', 'pres_lag_5', 'pres_lag_6', 'pres_lag_7', 'pres_lag_8', 'pres_lag_9', 'pres_lag_10', 'pres_lag_11', 'pres_lag_12'
]
X = data[features]
y = data[['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)), 
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)),
    tf.keras.layers.Dense(y_train.shape[1])              # Output layer with 7 neurons
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=16, validation_split=0.2)
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
#---------------------------------------------------------------------------------------------------------------------------------------
day_data = data[(data['month'] == 10) & (data['day'] == 27)]

# Calculate average values for each feature on October 30
historical_averages = day_data[['tavg','tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']].mean()

# Create a dictionary with the historical averages for October 30
input_data = {
    'tavg': historical_averages['tavg'],
    'tmin': historical_averages['tmin'],
    'tmax': historical_averages['tmax'],
    'prcp': historical_averages['prcp'],
    'wdir': historical_averages['wdir'],
    'wspd': historical_averages['wspd'],
    'pres': historical_averages['pres'],
    'month': 10,  # October
    'day': 27,     # Day 30
}

# Add historical average lags
for feature in ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']:
    for lag in range(1, lags + 1):
        input_data[f'{feature}_lag_{lag}'] = historical_averages[feature]
# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])[features]
print("Input shape:", input_df.shape)
new_predicted = model.predict(input_df)[0]


def c_to_f(degree):
    return degree * (9/5) + 32


result = get_data('2019-10-27')
print(' from tensor 10-27-19 model')
print(f"tavg: {c_to_f(new_predicted[0]):.2f} °F")
print(f"tmin: {c_to_f(new_predicted[1]):.2f} °F")
print(f"tmax: {c_to_f(new_predicted[2]):.2f} °F")
print(f"prcp: {new_predicted[3]:.2f} mm")
print(f"wdir: {new_predicted[4]:.2f} degrees")
print(f"wspd: {new_predicted[5]:.2f} km/h")
print(f"pres: {new_predicted[6]:.2f} hPa")
perc_error(abs_error(c_to_f(result['tavg']), c_to_f(new_predicted[0])), c_to_f(new_predicted[0]))
perc_error(abs_error(c_to_f(result['tmin']), c_to_f(new_predicted[1])), c_to_f(new_predicted[1]))
perc_error(abs_error(c_to_f(result['tmax']), c_to_f(new_predicted[2])), c_to_f(new_predicted[2]))
