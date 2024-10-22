import requests
import json
import prettytable
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

def get_data_from_bls(seriesID,startYr,endYr,bls_key,industry_name):
    headers = {'Content-type': 'application/json'}
    bls_api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    data = json.dumps({
        "seriesid": [seriesID],
        "startyear": startYr,
        "endyear": endYr,
        "registrationKey": bls_key
    })

    series_list = []

    p = requests.post(bls_api_url, data=data, headers=headers)

    if p.status_code != 200:
        raise Exception(f"Error: {p.status_code}, {p.text}")

    json_data = json.loads(p.text)

    for series in json_data['Results']['series']:
        seriesId = series['seriesID']
        for item in series['data']:
            year = item['year']
            period = item['period']
            value = item['value']

            if 'M01' <= period <= 'M12':
                series_list.append([seriesId, industry_name, year, period, value])

    return series_list

def train_model_and_predict():
    # Sample Time Series Data: Generate a DataFrame for multiple professions
    date_range = pd.date_range(start='2020-01-01', periods=24, freq='M')

    # Sample professions
    professions = ['Software Engineer', 'Nurse', 'Data Scientist', 'Sales Manager', 'Teacher']
    data = {
        'Date': np.tile(date_range, len(professions)),
        'Profession': np.repeat(professions, len(date_range)),
        'Job_Postings': np.random.randint(1000, 5000, size=(len(professions) * len(date_range),)),
        'Growth_Rate': np.random.uniform(0.01, 0.2, size=(len(professions) * len(date_range),))
    }

    df = pd.DataFrame(data)

    # Calculate Hot Profession Score
    df['Hot_Profession_Score'] = df['Job_Postings'] * df['Growth_Rate']

    # Set the date as the index
    df.set_index('Date', inplace=True)

    # Normalize and prepare data for each profession
    def create_datasets_for_professions(df, profession, time_step=1):
        profession_data = df[df['Profession'] == profession][['Hot_Profession_Score']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(profession_data)

        X, y = [], []
        for i in range(len(scaled_data) - time_step - 1):
            X.append(scaled_data[i:(i + time_step), 0])
            y.append(scaled_data[i + time_step, 0])
        return np.array(X), np.array(y), scaler, profession_data

    # Train and predict for each profession
    for profession in professions:
        X, y, scaler, profession_data = create_datasets_for_professions(df, profession, time_step=3)

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build the LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.LSTM(25),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform the predictions to original scale
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Prepare data for plotting
        plt.figure(figsize=(12, 6))
        plt.plot(profession_data.index, profession_data['Hot_Profession_Score'], label='Actual Score', color='blue')
        train_indices = profession_data.index[3:3 + len(train_predict)]
        test_indices = profession_data.index[3 + len(train_predict) + 1:]

        plt.plot(train_indices, train_predict, label='Train Predictions', color='orange')
        plt.plot(test_indices, test_predict, label='Test Predictions', color='red')
        plt.title(f'{profession} Hot Profession Score Forecast')
        plt.xlabel('Date')
        plt.ylabel('Hot Profession Score')
        plt.legend()
        plt.show()

        # Print model summary
        model.summary()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Team 90')
    # Sample list of BLS Series IDs for pulling Employment data
    series_data = [
        ['CEU0000000001', 'Total nonfarm'],
        ['CEU0500000001', 'Total private'],
        ['CEU0600000000001', 'Goods-producing'],
        ['CEU0700000000001', 'Service-providing'],
        ['CEU4000000000001', 'Trade, transportation, and utilities'],
        ['CEU4142000000001', 'Wholesale trade'],
        ['CEU4200000000001', 'Retail trade'],
        ['CEU6056150000001', 'Travel arrangement and reservation services'],
        ['CEU7072250000001', 'Restaurants and other eating places'],
        ['CEU7071000000001', 'Arts, entertainment, and recreation'],
        ['CEU6054150000001', 'Computer systems design and related services'],
        ['CEU6054161000001', 'Management consulting services']
    ]

    # Create a DataFrame
    df_series = pd.DataFrame(series_data, columns=["seriesID", "industry_name"])
    print(df_series)

    # Setup parameters
    startYr = '2020'
    endYr = '2024'
    bls_key = 'a2baff6ba61547918ea822dd565b8e55'  # Replace with your actual BLS API key

    # Initialize the list to capture the data
    complete_list = []

    try:
        for index, row in df_series.iterrows():
            print("Loading data for {}...".format(row['seriesID']))
            series_list = get_data_from_bls(row['seriesID'], startYr, endYr, bls_key, row['industry_name'])
            complete_list.extend(series_list)

    except Exception as e:
        print(e)

    # Create a DataFrame from the complete_list
    columns = ["seriesID", "industry_name", "year", "period", "value"]
    df_bls = pd.DataFrame(complete_list, columns=columns)

    # Display the DataFrame
    print(df_bls)
   # get_data_from_bls()
    #train_model_and_predict()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
