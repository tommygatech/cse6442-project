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
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
def train_model_and_predict():
    # Sample Time Series Data: Generate a DataFrame for multiple professions
    date_range = pd.date_range(start='2024-01-01', periods=24, freq='ME')
    number_of_month = 12
    # Sample professions and states
    professions = ['Software Engineer', 'Nurse', 'Data Scientist', 'Sales Manager', 'Teacher']
    states = ['California', 'Texas', 'New York', 'Florida', 'Illinois']

    # Generate random job postings and growth rates
    data = {
        'Date': np.tile(date_range, len(professions) * len(states)),
        'Profession': np.repeat(professions, len(date_range) * len(states)),
        'State': np.tile(states, len(professions) * len(date_range)),
        'Job_Postings': np.random.randint(1000, 5000, size=(len(professions) * len(states) * len(date_range),)),
        'Growth_Rate': np.random.uniform(0.01, 0.2, size=(len(professions) * len(states) * len(date_range),))
    }

    df = pd.DataFrame(data)

    # Calculate Hot Profession Score
    df['Hot_Profession_Score'] = df['Job_Postings'] * df['Growth_Rate']
    df.set_index('Date', inplace=True)

    # Store predictions
    predictions = []

    # Train and predict for each profession in each state
    for state in states:
        for profession in professions:
            # Filter data
            filtered_data = df[(df['Profession'] == profession) & (df['State'] == state)][['Hot_Profession_Score']]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(filtered_data)

            # Prepare the dataset
            time_step = 3
            X, y = [], []
            for i in range(len(scaled_data) - time_step - 1):
                X.append(scaled_data[i:(i + time_step), 0])
                y.append(scaled_data[i + time_step, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Create and train the LSTM model
            model = create_lstm_model(input_shape=(X.shape[1], 1))
            model.fit(X, y, epochs=100, batch_size=1, verbose=0)

            # Make predictions for the next 5 years (60 months)
            future_predictions = []
            last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)

            for _ in range(number_of_month):  # Predict 60 months ahead
                next_prediction = model.predict(last_sequence)
                future_predictions.append(next_prediction[0, 0])
                last_sequence = np.append(last_sequence[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Calculate the average predicted score
            avg_score = np.mean(future_predictions)
            predictions.append((state, profession, avg_score))

    # Create a DataFrame for predictions and find the top 5 hottest jobs for each state
    predictions_df = pd.DataFrame(predictions, columns=['State', 'Profession', 'Avg_Score'])

    # Sort and get the top 5 hottest jobs for each state
    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score'], ascending=[True, False])
    top_5_jobs = top_hottest_jobs.groupby('State').head(3)

    # Save the results to a CSV file
    top_5_jobs.to_csv('hottest_jobs_by_state.csv', index=False)
    print("Top 5 hottest jobs for each state saved to 'hottest_jobs_by_state.csv'.")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Team 90')
    # Sample list of BLS Series IDs for pulling Employment data
    '''
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
    '''
   # get_data_from_bls()
    train_model_and_predict()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
