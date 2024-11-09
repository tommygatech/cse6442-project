# pip install pandas openpyxl
import requests
import json
import os
import shutil

import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Team 90')

    for n in range(12, 61, 12):
        df_bls_combined = pd.read_csv('professionsoutput.csv')
        train_model_and_predict(df_bls_combined, n)

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
