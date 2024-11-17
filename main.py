#pip install pandas openpyxl
#pip install numpy
#pip install requests
#pip install scikit-learn
#pip install pmdarima
#pip install tqdm
import sys
import subprocess
import os
import shutil
import warnings
import json
import requests
import time
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

library_name = 'numpy'
if library_name in sys.modules:
    print("NumPy is already imported!")
else:
    # Install the library if not imported
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    import numpy as np

library_name = 'pandas'
if library_name in sys.modules:
    print("pandas is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    import pandas as pd
library_name = 'scikit-learn'
if library_name in sys.modules:
    print("scikit-learn is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    from sklearn.model_selection import (train_test_split,RandomizedSearchCV, TimeSeriesSplit)
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import (PolynomialFeatures, StandardScaler,OneHotEncoder,MinMaxScaler)
    from sklearn.metrics import (mean_squared_error, make_scorer)
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

library_name = 'statsmodels'
if library_name in sys.modules:
    print("statsmodels is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    from statsmodels.tsa.arima.model import ARIMA
library_name = 'pmdarima'
if library_name in sys.modules:
    print("auto_arima is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    from pmdarima import auto_arima

library_name = 'tqdm'
if library_name in sys.modules:
    print("tqdm is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    from tqdm import tqdm


def print_hi(name):
    print(f'Hi {name}')

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

def train_model_and_predict_LinearRegression(data, number_of_months, max_lag=12):
    # Set index and extract unique states
    #data.set_index('seriesID', inplace=True)

    states = data['state'].unique()
    professions = data['industry_name'].unique()
    data = data.drop('seriesID', axis=1)
    data = data.sort_values(by=['state', 'industry_name', 'year', 'period'])
    data['Hot_Profession_Score'] = data['value']

    predictions = []

    for state in states:
        for profession in professions:
            group_data = data[(data['state'] == state) & (data['industry_name'] == profession)].copy()
            if group_data.empty:
                continue

            # Create lagged variables for 'value' column
            group_data = create_lags(group_data, max_lag)
            lag_columns = [f'Lag_{i}' for i in range(1, max_lag + 1)]
            group_data.dropna(inplace=True)  # Drop any rows with missing values

            if group_data.empty or len(group_data) < max_lag:
                continue

            X = group_data[lag_columns].values
            y = group_data['Hot_Profession_Score'].values

            # Initialize the Ridge regression model
            model = Ridge(alpha=10)
            model.fit(X, y)

            # Start with the last available lags for prediction
            last_known_lags = X[-1].reshape(1, -1)
            avg_score = model.predict(last_known_lags)[0]
            future_predictions = {}

            for month in range(1, number_of_months + 1):  # Predict for up to the maximum months
                next_month_lags = np.roll(last_known_lags.copy(), -1, axis=1)
                next_month_lags[0, -1] = avg_score

                try:
                    avg_score = model.predict(next_month_lags)[0] # Predict next month
                except ValueError as e:
                    print(f"Error predicting for state: {state}, profession: {profession}, month: {month}: {e}")
                    break

                # Avoid large values by clipping if necessary (to avoid overflow)
                avg_score = np.clip(avg_score, -1e10, 100000)

                # Store prediction only if it's in the desired future periods
                if month == number_of_months:
                    future_predictions[month] = avg_score

                last_known_lags = next_month_lags

            # Store the prediction for the current profession and state
            for month_ahead, prediction in future_predictions.items():
                predictions.append((state, profession, prediction))

    # Sort and get the top 10 hottest jobs for each state
    predictions_df = pd.DataFrame(predictions, columns=['State', 'Profession', 'Avg_Score'])
    predictions_df['Avg_Score'] = predictions_df['Avg_Score'].round(2) # Round to 2 decimal places for readability

    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score'], ascending=[True, False])
    top_10_jobs = top_hottest_jobs.groupby('State').head(10)

    # Save the results to a CSV file
    output_filename = f"{number_of_months}hottest_jobs_by_state.csv"
    top_10_jobs.to_csv(output_filename, index=False)
    print(f"Top 10 hottest jobs for each state saved to '{output_filename}'.")


def train_model_and_predict_with_different_algorithms(data, number_of_month, k=5):
    data.set_index('seriesID', inplace=True)
    states = data['state'].unique()
    professions = data['industry_name'].unique()

    data['Hot_Profession_Score'] = data['value'] * np.random.uniform(0.01, 0.2, size=len(data))

    scaler = StandardScaler()
    data['Hot_Profession_Score_scaled'] = scaler.fit_transform(data['Hot_Profession_Score'].values.reshape(-1, 1)).flatten()

    predictions = []
    mse_values_lr = []
    mse_values_rf = []
    mse_values_arima = []
    mse_values_ensemble = []

    for state in tqdm(states):
        for profession in professions:
            filtered_data = data[(data['industry_name'] == profession) & (data['state'] == state)][
                ['Hot_Profession_Score_scaled']]
            if filtered_data.empty:
                continue

            X = np.arange(len(filtered_data)).reshape(-1, 1)
            y = filtered_data['Hot_Profession_Score_scaled'].values

            kf = KFold(n_splits=k, shuffle=True, random_state=13)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                poly = PolynomialFeatures(degree=2)
                X_poly_train = poly.fit_transform(X_train)
                X_poly_test = poly.transform(X_test)

                model_lr = LinearRegression()
                model_lr.fit(X_poly_train, y_train)
                y_pred_lr = model_lr.predict(X_poly_test)
                mse_lr = mean_squared_error(y_test, y_pred_lr)
                mse_values_lr.append(mse_lr)

                model_rf = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=3)
                model_rf.fit(X_train, y_train)
                y_pred_rf = model_rf.predict(X_test)
                mse_rf = mean_squared_error(y_test, y_pred_rf)
                mse_values_rf.append(mse_rf)

                try:
                    model_arima = auto_arima(y_train, start_p=1, start_d=1, start_q=1)
                    model_arima_fit = model_arima.fit(y_train)
                    y_pred_arima = model_arima_fit.predict(len(y_test))
                    mse_arima = mean_squared_error(y_test, y_pred_arima)
                    mse_values_arima.append(mse_arima)
                except Exception as e:
                    print(f"Error in ARIMA: {e}")
                    y_pred_arima = np.full(len(y_test), np.nan)

                try:
                    y_pred_ensemble = (y_pred_lr + y_pred_rf + y_pred_arima) / 3
                except Exception as e:
                    print(f"Error in Ensemble: {e}")
                    y_pred_ensemble = (y_pred_lr + y_pred_rf) / 2

                mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
                mse_values_ensemble.append(mse_ensemble)

                pred_months = np.arange(len(filtered_data), len(filtered_data) + number_of_month).reshape(-1, 1)
                pred_months_poly = poly.transform(pred_months)
                y_pred_lr_next = model_lr.predict(pred_months_poly)
                y_pred_rf_next = model_rf.predict(pred_months)
                try:
                    y_pred_arima_next = model_arima_fit.predict(len(y_test) + number_of_month)
                    y_pred_arima_next = y_pred_arima_next[-number_of_month:]
                except Exception as e:
                    print(f"Error in ARIMA forecast: {e}")
                    y_pred_arima_next = np.full(number_of_month, np.nan)
                try:
                    y_pred_ensemble_next = (y_pred_lr_next + y_pred_rf_next + y_pred_arima_next) / 3
                except Exception as e:
                    print(f"Error in Ensemble forecast: {e}")
                    y_pred_ensemble_next = (y_pred_lr_next + y_pred_rf_next) / 2

                avg_score_lr = np.mean(y_pred_lr_next)
                avg_score_rf = np.mean(y_pred_rf_next)
                avg_score_arima = np.mean(y_pred_arima_next)
                avg_score_ensemble = np.mean(y_pred_ensemble_next)
                predictions.append((state, profession, avg_score_lr, avg_score_rf, avg_score_arima, avg_score_ensemble))

    predictions_df = pd.DataFrame(predictions, columns=['State', 'Profession', 'Avg_Score_LR', 'Avg_Score_RF', 'Avg_Score_ARIMA', 'Avg_Score_Ensemble'])
    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score_Ensemble'], ascending=[True, False])
    top_10_jobs = top_hottest_jobs.groupby('State').head(10)

    output_filename = f"{number_of_month}hottest_jobs_by_state.csv"
    top_10_jobs.to_csv(output_filename, index=False)

    avg_mse_lr = np.mean(mse_values_lr)
    avg_mse_rf = np.mean(mse_values_rf)
    avg_mse_arima = np.mean(mse_values_arima)
    avg_mse_ensemble = np.mean(mse_values_ensemble)

    print(f"Average MSE (Linear Regression): {avg_mse_lr:.2f}")
    print(f"Average MSE (Random Forest): {avg_mse_rf:.2f}")
    print(f"Average MSE (ARIMA): {avg_mse_arima:.2f}")
    print(f"Average MSE (Ensemble): {avg_mse_ensemble:.2f}")

    models = ['Linear Regression', 'Random Forest', 'ARIMA', 'Ensemble']
    mse_values = [avg_mse_lr, avg_mse_rf, avg_mse_arima, avg_mse_ensemble]
    best_model = models[np.argmin(mse_values)]
    print(f"Best Model: {best_model}")
def train_model_and_predict_RandomForest(data, number_of_months, max_lag=12):
    states = data['state'].unique()
    professions = data['industry_name'].unique()
    data = data.drop('seriesID', axis=1)
    data = data.sort_values(by=['state', 'industry_name', 'year', 'period'])
    data['Hot_Profession_Score'] = data['value']

    # Store predictions
    predictions = []
    data_grouped_by_year = data.groupby(['state', 'industry_name', 'year']).agg({
        'value': 'sum',
        'Hot_Profession_Score': 'sum'
    }).reset_index()
    s=0
    for state in states:
        s = s + 1
        state_max_lag = max_lag + s
        # Create lagged variables for 'value' column
        data_with_lags = create_lags(data_grouped_by_year.copy(), state_max_lag)
        lag_columns = [f'Lag_{i}' for i in range(1, state_max_lag + 1)]
        data_with_lags.dropna(inplace=True)  # Drop any rows with missing values

        X = data_with_lags[lag_columns].values
        y = data_with_lags['Hot_Profession_Score'].values

        # Initialize the model and PolynomialFeatures
        poly = PolynomialFeatures(degree=2)
        X_poly_train = poly.fit_transform(X)
        # Initialize RandomForestRegressor
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_poly_train, y)
        for profession in professions:
            filtered_data = data_with_lags[
                (data_with_lags['industry_name'] == profession) & (data_with_lags['state'] == state)]
            print(f'state:{state} - profession:{profession}')
            if filtered_data.empty:
                continue

            # Sort the data by year (ascending order)
            filtered_data = filtered_data.sort_values(by='year', ascending=False)
            last_year_data = filtered_data.iloc[0]
            pred_lags = last_year_data[lag_columns].values.reshape(1, -1)
            pred_months_poly = poly.transform(pred_lags)
            avg_score = model.predict(pred_months_poly)[0]

            for years_ahead in range(1, number_of_months + 1):  # Predict for 1 to 'number_of_years' years ahead
                months_ahead = years_ahead * 12  # Convert years to months
                for month in range(months_ahead):
                    future_pred_lags = np.roll(pred_lags.copy(), shift=-1, axis=1)
                    future_pred_lags[0, -1] = avg_score
                    future_pred_lags = future_pred_lags.astype(np.float64)

                    if not np.all(np.isfinite(future_pred_lags)):
                        print(f"Warning: NaN or Inf detected in future predictions. Skipping this prediction.")
                        break
                    future_pred_poly = poly.transform(future_pred_lags)
                    try:
                        avg_score = model.predict(future_pred_poly)[0]
                    except ValueError as e:
                        print(
                            f"Error predicting for state: {state}, profession: {profession}, year_ahead: {years_ahead}: {e}")
                        continue

                    avg_score = np.clip(avg_score, -1e10, 100000)  # Limit the values to a reasonable range

                # Store the prediction for the current profession and state
                predictions.append((state, profession, avg_score))

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions, columns=['State', 'Profession', 'Avg_Score'])

    # Remove duplicate predictions (if any), based on State, Profession, and Year
    predictions_df = predictions_df.drop_duplicates(subset=['State', 'Profession'])

    # Sort and get the top 10 hottest jobs for each state
    predictions_df['Avg_Score'] = predictions_df['Avg_Score'].round(2)  # Round to 2 decimal places for readability
    top_hottest_jobs = predictions_df.sort_values(by=['State', 'Avg_Score'], ascending=[True, False])
    top_10_jobs = top_hottest_jobs.groupby('State').head(10)

    if number_of_months < 12:
        number_of_months = number_of_months * 12

    # Save the results to a CSV file
    output_filename = f"{number_of_months}hottest_jobs_by_state.csv"
    top_10_jobs.to_csv(output_filename, index=False)
    print(f"Top 10 hottest jobs for each state saved to '{output_filename}'.")
def create_lags(data_input, max_lag):
    for lag in range(1, max_lag + 1):
        data_input[f'Lag_{lag}'] = np.roll(data_input['value'], lag)
    return data_input
def get_series_data():
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
    return series_data
def copy_files():
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
            # Adjust the condition as necessary based on your data_input
            filtered_data = filtered_data[~filtered_data[col_a].astype(str).str.contains('Footnote', na=False)]
            filtered_data = filtered_data[~filtered_data[col_a].astype(str).str.contains('Footnote', na=False)]
            filtered_data = filtered_data.dropna(subset=[col_a, col_e])
            filtered_data = filtered_data[[col_a, col_e]]  # Select only the desired columns
            print(f"Contents of Sheet {index + 1}:")
            print(filtered_data)
            allProfessions.append(filtered_data)
            print()  # Add a newline for better readability


    series_data = get_series_data()
    # Create a DataFrame
    df_series = pd.DataFrame(series_data, columns=["seriesID", "industry_name", "state"])
    print(df_series)

    # Setup parameters
    startYr = '2020'
    endYr = '2024'
    bls_key = 'a2baff6ba61547918ea822dd565b8e55'  # Replace with your actual BLS API key

    # Initialize the list to capture the data_input
    complete_list = []
    print("invoking api section")
    try:
        for index, row in df_series.iterrows():
            print("Loading data_input for {}...".format(row['seriesID']))
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
    # Check if the DataFrame has data before saving to a file
    if not df_bls_combined.empty:
        df_bls_combined.to_csv('professionsoutput.csv', index=False)  # Save to CSV (or another file format)
    else:
        print("DataFrame is empty. Nothing to save.")
    algorithm = "LR"
    dataInput = pd.read_csv('professionsoutput.csv')
    max_lag=9
    for n in range(12, 61, 12):
        if algorithm == "LR":
            train_model_and_predict_LinearRegression(dataInput.copy(), n, max_lag=max_lag)
        else:
            train_model_and_predict_RandomForest(dataInput.copy(), n, max_lag=max_lag)
    copy_files()
