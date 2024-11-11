#pip install pandas openpyxl
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
library_name = 'numpy'
if library_name in sys.modules:
    print("NumPy is already imported!")
else:
    # Install the library if not imported
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    import numpy as np
library_name = 'requests'
if library_name in sys.modules:
    print("requests is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])


library_name = 'pandas'
if library_name in sys.modules:
    print("pandas is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    import pandas as pd
library_name = 'tensorflow'
if library_name in sys.modules:
    print("tensorflow is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    import tensorflow as tf
library_name = 'scikit-learn'
if library_name in sys.modules:
    print("scikit-learn is already imported!")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import (PolynomialFeatures, StandardScaler)
    from sklearn.metrics import mean_squared_error
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
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

def train_model_and_predict(data, number_of_month):
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

            model.fit(X_poly_train, y_train)
            y_pred = model.predict(X_poly_test)

            y_test = y_test.flatten()  # Make sure y_test is 1D
            y_pred = y_pred.flatten()  # Make sure y_pred is 1D

            # Calculate MSE for this profession in this state
            mse = mean_squared_error(y_test, y_pred)
            print(f"MSE for {state} - {profession}: {mse}")

            # Predict future months (beyond the training data)
            pred_months = np.arange(len(filtered_data), len(filtered_data) + number_of_month).reshape(-1, 1)
            pred_months_poly = poly.transform(pred_months)
            predictions_array = model.predict(pred_months_poly)
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
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Team 90')

    for n in range(12, 61, 12):
        df_bls_combined = pd.read_csv('professionsoutput.csv')
        #train_model_and_predict_with_different_algorithms(df_bls_combined, n, k=4)
        train_model_and_predict(df_bls_combined,n)
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
