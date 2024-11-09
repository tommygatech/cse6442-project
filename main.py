#pip install pandas openpyxl
import requests
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import shutil

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

def get_data_from_bls(seriesID,startYr,endYr,bls_key,industry_name,state):
    headers = {'Content-type': 'application/json'}
    bls_api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    series_ids = seriesID.split(',')

    # Wrap each ID in double quotes and join with commas
    data = json.dumps({
        "seriesid": series_ids,
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
            periodName = item['periodName']

            if 'M01' <= period <= 'M12':
                series_list.append([seriesId,state, industry_name, year, periodName, value])

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
def train_model_and_predict(data,number_of_month):

    data.set_index('seriesID', inplace=True)
    states = data['state'].unique()
    professions = data['industry_name'].unique()
    data['Hot_Profession_Score'] = data['value'] * np.random.uniform(0.01, 0.2, size=len(data))
    # Store predictions
    predictions = []
    future_predictions = []
    # Train and predict for each profession in each state
    for state in states:
        for profession in professions:
            # Filter data
            filtered_data = data[(data['industry_name'] == profession) & (data['state'] == state)][['Hot_Profession_Score']]
            #filtered_data = data[(data['state'] == state)][['Hot_Profession_Score']]
            if filtered_data.empty:
                continue
            '''lst'''
            # Prepare the dataset

            time_step = 3
            X, y = [], []
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(filtered_data)
            for i in range(len(scaled_data) - time_step - 1):
                X.append(scaled_data[i:(i + time_step), 0])
                y.append(scaled_data[i + time_step, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Create and train the LSTM model
            model = create_lstm_model(input_shape=(X.shape[1], 1))
            model.fit(X, y, epochs=5, batch_size=1, verbose=0)

            # Make predictions for the next 5 years (60 months)

            last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)

            for _ in range(number_of_month):  # Predict 60 months ahead
                next_prediction = model.predict(last_sequence)
                #next_prediction = model.predict(X_test)
                future_predictions.append(next_prediction[0, 0])
                last_sequence = np.append(last_sequence[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


    predictions_df = pd.DataFrame(future_predictions, columns=['State', 'Profession', 'Avg_Score'])

    # Sort and get the top 5 hottest jobs for each state
    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score'], ascending=[True, False])
    top_5_jobs = top_hottest_jobs.groupby('State').head(10)


    # Save the results to a CSV file
    top_5_jobs.to_csv(f"{number_of_month}_hottest_jobs_by_state.csv", index=False)
    print("Top 5 hottest jobs for each state saved to 'hottest_jobs_by_state.csv'.")

def train_model_and_predict_LinearRegression(data, number_of_month):
    # Set index and extract unique states and professions
    data.set_index('seriesID', inplace=True)
    states = data['state'].unique()
    professions = data['industry_name'].unique()

    # Calculate Hot Profession Score
    data['Hot_Profession_Score'] = data['value'] * np.random.uniform(0.01, 0.2, size=len(data))

    # Store predictions
    predictions = []

    # Train and predict for each profession in each state
    for state in states:
        for profession in professions:
            # Filter data
            filtered_data = data[(data['industry_name'] == profession) & (data['state'] == state)][
                ['Hot_Profession_Score']]
            if filtered_data.empty:
                continue

            # Prepare the dataset
            X = np.arange(len(filtered_data)).reshape(-1, 1)  # Time as feature
            y = filtered_data['Hot_Profession_Score'].values

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
            # Use Polynomial Features
            poly = PolynomialFeatures(degree=2)
            X_poly_train = poly.fit_transform(X_train)
            X_poly_test = poly.transform(X_test)
            model = LinearRegression()

            # Fit the model to the training data
            model.fit(X_poly_train, y_train)
            pred_months = np.arange(len(filtered_data), len(filtered_data) + number_of_month).reshape(-1, 1)
            pred_months_poly = poly.transform(pred_months)
            predictions_array = model.predict(pred_months_poly)
            #predictions_array = model.predict(pred_months)

            # Calculate the average predicted score
            avg_score = np.mean(predictions_array)
            predictions.append((state, profession, avg_score))

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions, columns=['State', 'Profession', 'Avg_Score'])

    # Sort and get the top 10 hottest jobs for each state
    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score'], ascending=[True, False])
    top_10_jobs = top_hottest_jobs.groupby('State').head(10)

    # Save the results to a CSV file
    output_filename = f"{number_of_month}hottest_jobs_by_state.csv"
    top_10_jobs.to_csv(output_filename, index=False)
    print(f"Top 10 hottest jobs for each state saved to '{output_filename}'.")

if __name__ == '__main__':
    print_hi('Team 90')
    # cleanData
    sheets_array = []
    sheets = pd.read_excel('occupation.xlsx', sheet_name=None,header=None)
    # Loop through each sheet and print its content
    for sheet_name, df in sheets.items():
        sheets_array.append(df)
    allProfessions=[]
    for index, df in enumerate(sheets_array):
        if 2 < index < 6:
            column_names = df.columns
            col_a = column_names[0]
            col_e = column_names[5]  # 'Unnamed: 4'
            # Filter the DataFrame
            # Exclude footnotes based on a condition (e.g., exclude rows that contain "Footnote")
            filtered_data = df.iloc[3:]  # Start from the 4th row

            # Example condition to exclude footnotes
            # Adjust the condition as necessary based on your data
            filtered_data = filtered_data[~filtered_data[col_a].astype(str).str.contains('Footnote', na=False)]
            filtered_data = filtered_data[~filtered_data[col_a].astype(str).str.contains('Footnote', na=False)]
            filtered_data = filtered_data.dropna(subset=[col_a, col_e])
            filtered_data = filtered_data[[col_a, col_e]]  # Select only the desired columns
            print(f"Contents of Sheet {index + 1}:")
            print(filtered_data)
            allProfessions.append(filtered_data)
            print()  # Add a newline for better readability


    series_data = [
        ['JTS000000020000000JOR', 'Total nonfarm', 'Alabama'],
        ['JTS000000020000000JOR', 'Total nonfarm', 'Alaska'],
        ['JTS000000040000000JOR', 'Total nonfarm', 'Arizona'],
        ['JTS000000050000000JOR', 'Total nonfarm', 'Arkansas'],
        ['JTS000000060000000JOR', 'Total nonfarm', 'California'],
        ['JTS000000080000000JOR', 'Total nonfarm', 'Colorado'],
        ['JTS000000090000000JOR', 'Total nonfarm', 'Connecticut'],
        ['JTS000000100000000JOR', 'Total nonfarm', 'Delaware'],
        ['JTS000000110000000JOR', 'Total nonfarm', 'District of Columbia'],
        ['JTS000000120000000JOR', 'Total nonfarm', 'Florida'],
        ['JTS000000130000000JOR', 'Total nonfarm', 'Georgia'],
        ['JTS000000150000000JOR', 'Total nonfarm', 'Hawaii'],
        ['JTS000000160000000JOR', 'Total nonfarm', 'Idaho'],
        ['JTS000000170000000JOR', 'Total nonfarm', 'Illinois'],
        ['JTS000000180000000JOR', 'Total nonfarm', 'Indiana'],
        ['JTS000000190000000JOR', 'Total nonfarm', 'Iowa'],
        ['JTS000000200000000JOR', 'Total nonfarm', 'Kansas'],
        ['JTS000000210000000JOR', 'Total nonfarm', 'Kentucky'],
        ['JTS000000220000000JOR', 'Total nonfarm', 'Louisiana'],
        ['JTS000000230000000JOR', 'Total nonfarm', 'Maine'],
        ['JTS000000240000000JOR', 'Total nonfarm', 'Maryland'],
        ['JTS000000250000000JOR', 'Total nonfarm', 'Massachusetts'],
        ['JTS000000260000000JOR', 'Total nonfarm', 'Michigan'],
        ['JTS000000270000000JOR', 'Total nonfarm', 'Minnesota'],
        ['JTS000000280000000JOR', 'Total nonfarm', 'Mississippi'],
        ['JTS000000290000000JOR', 'Total nonfarm', 'Missouri'],
        ['JTS000000300000000JOR', 'Total nonfarm', 'Montana'],
        ['JTS000000310000000JOR', 'Total nonfarm', 'Nebraska'],
        ['JTS000000320000000JOR', 'Total nonfarm', 'Nevada'],
        ['JTS000000330000000JOR', 'Total nonfarm', 'New Hampshire'],
        ['JTS000000340000000JOR', 'Total nonfarm', 'New Jersey'],
        ['JTS000000350000000JOR', 'Total nonfarm', 'New Mexico'],
        ['JTS000000360000000JOR', 'Total nonfarm', 'New York'],
        ['JTS000000370000000JOR', 'Total nonfarm', 'North Carolina'],
        ['JTS000000380000000JOR', 'Total nonfarm', 'North Dakota'],
        ['JTS000000390000000JOR', 'Total nonfarm', 'Ohio'],
        ['JTS000000400000000JOR', 'Total nonfarm', 'Oklahoma'],
        ['JTS000000410000000JOR', 'Total nonfarm', 'Oregon'],
        ['JTS000000420000000JOR', 'Total nonfarm', 'Pennsylvania'],
        ['JTS000000440000000JOR', 'Total nonfarm', 'Rhode Island'],
        ['JTS000000450000000JOR', 'Total nonfarm', 'South Carolina'],
        ['JTU000000460000000JOR', 'Total nonfarm', 'South Dakota'],
        ['JTS000000470000000JOR', 'Total nonfarm', 'Tennessee'],
        ['JTS000000480000000JOR', 'Total nonfarm', 'Texas'],
        ['JTS000000490000000JOR', 'Total nonfarm', 'Utah'],
        ['JTS000000500000000JOR', 'Total nonfarm', 'Vermont'],
        ['JTS000000510000000JOR', 'Total nonfarm', 'Virginia'],
        ['JTS000000530000000JOR', 'Total nonfarm', 'Washington'],
        ['JTS000000540000000JOR', 'Total nonfarm', 'West Virginia'],
        ['JTS000000550000000JOR', 'Total nonfarm', 'Wisconsin'],
        ['JTS000000560000000JOR', 'Total nonfarm', 'Wyoming']
    ]
    # Create a DataFrame
    df_series = pd.DataFrame(series_data, columns=["seriesID", "industry_name", "state"])
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
            series_list = get_data_from_bls(row['seriesID'], startYr, endYr, bls_key, row['industry_name'],row['state'])
            complete_list.extend(series_list)

    except Exception as e:
        print(e)

    # Create a DataFrame from the complete_list
    columns = ["seriesID","state", "industry_name", "year", "period", "value"]
    df_bls = pd.DataFrame(complete_list, columns=columns)
    
    # Display the DataFrame
    #print(df_bls)
    df_bls.to_csv('output.csv', index=False)
   # get_data_from_bls()
    #df_bls=pd.read_csv('output.csv')
    combinedData=[]
    for index in allProfessions:
        for d in index.values:
            industry_name = d[0]
            value_multiplier = d[1]
            for dix, row in df_bls.iterrows():
                # Create a new row for each entry in df_bls
                new_row = {
                    "seriesID": row["seriesID"],
                    "state": row["state"],
                    "industry_name": industry_name,
                    "year": row["year"],
                    "period": row["period"],
                    "value": float(row["value"]) * float(value_multiplier)  # Multiply the value by the multiplier
                }
                combinedData.append(new_row)
    #columns = ["seriesID", "state", "industry_name", "year", "period", "value"]
    df_bls_combined = pd.DataFrame(combinedData, columns=columns)
    df_bls_combined.to_csv('professionsoutput.csv', index=False)
    for n in range(12, 61, 12):
        train_model_and_predict_LinearRegression(df_bls_combined,n)
        df_bls_combined =pd.read_csv('professionsoutput.csv')
    source_folder = os.getcwd()
    target_folder = os.path.join(source_folder, 'UI')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate through all files in the current working directory
    for filename in os.listdir(source_folder):
        # Check if '_state' is in the filename
        if 'jobs_by_state' in filename:
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(source_path):
                # Copy the file to the target folder
                shutil.copy(source_path, target_path)
                print(f"Copied: {filename}")

    print("File copying complete.")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
