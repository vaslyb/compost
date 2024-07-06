import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os
import statsmodels.api as sm  
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Create a results directory if it does not exist
results_directory = './results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Load CSV file using Pandas
df = pd.read_csv('compost.csv')
df = df.replace(',', '.', regex=True).astype(float)

# Define input and output columns
input_indices = [1, 2, 3, 4, 5, 6]  # Adjusted to zero-based indexing
output_indices = [7, 8, 9, 10, 11]   # Adjusted to zero-based indexing

input_feature_names = df.columns[input_indices]
output_feature_names = df.columns[output_indices]
features = df.iloc[:, input_indices].values
target = df.iloc[:, output_indices].values
target_labels = df.columns[output_indices]

# Collect the test set in order to have representative data
selected_indices = [5, 10, 15, 20, 25, 30, 35, 40]
mask = np.zeros(features.shape[0], dtype=bool)
mask[selected_indices] = True

x_train = features[~mask]
x_test = features[mask]

target = target.astype(float)
y_train = target[~mask]
y_test = target[mask]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define regression models
models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=RANDOM_SEED),
    MultiOutputRegressor(LinearSVR(random_state=RANDOM_SEED)),
    XGBRegressor(random_state=RANDOM_SEED)
]

model_names = ['Linear Regression', 'K-Neighbors Regressor', 'Decision Tree Regressor', 'Support Vector Regressor', 'XGBoost Regressor']

results = {}
predictions_df = pd.DataFrame()

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for num, model in enumerate(models):
    model_name = type(model).__name__
    if isinstance(model, LinearRegression):
        x_train_with_intercept = sm.add_constant(x_train)  # Add intercept term to features

        # Calculate p-values for each target separately
        p_values_list = []
        for i in range(y_train.shape[1]):  # Assuming y_train has shape (n_samples, 5)
            y_train_1d = y_train[:, i]
            model_with_intercept = sm.OLS(y_train_1d, x_train_with_intercept).fit()
            p_values_list.append(model_with_intercept.pvalues)

        # Combine p-values into a DataFrame
        p_values_df = pd.DataFrame(p_values_list, index=output_feature_names)
        p_values_df = p_values_df.transpose()
        p_values_df.insert(0, 'Feature', ['Intercept'] + list(input_feature_names))

        # Save p-values to CSV file
        p_values_file_path = f'./results/{model_name}_p_values.csv'
        p_values_df.to_csv(p_values_file_path, index=False)
        
    # Train the model
    model.fit(x_train, y_train)
    
    # Save the coefficients to a CSV file
    if hasattr(model, 'intercept_'):
        coefficients = model.coef_
        intercept = model.intercept_
        coefficients = np.round(coefficients, 3)
        intercept = np.round(intercept, 3)
        
        # Store coefficients in a DataFrame
        coefficients_df = pd.DataFrame(coefficients, columns=input_feature_names)
        coefficients_df.insert(0, 'Output', output_feature_names)
        coefficients_df.insert(0, 'Model', model_name)
        coefficients_df['Intercept'] = intercept

        # Save coefficients to CSV file
        coefficients_file_path = f'./results/{model_name}_coefficients.csv'
        coefficients_df.to_csv(coefficients_file_path, index=False)
        
    # Visualize feature importance for XGBoost
    if isinstance(model, XGBRegressor):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  # 10x8 inches figure with 300 dpi resolution
        
        feature_importances = model.feature_importances_
        
        # Plot the feature importances
        sorted_idx = np.argsort(feature_importances)
        feature_names = np.array(input_feature_names)

        # Create a larger figure with higher resolution
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  # 10x8 inches figure with 300 dpi resolution

        # Plot feature importances
        bars = ax.barh(range(x_test.shape[1]), feature_importances[sorted_idx], color='red', height=0.5)

        # Set the title and increase the font size of y and x labels
        plt.title('XGBoost Feature Importance', fontsize=16)
        ax.set_ylabel('Features', fontsize=14)
        ax.set_xlabel('Score', fontsize=14)
        # Increase the font size of y-tick labels and set feature names
        ax.set_yticks(np.arange(x_test.shape[1]))
        ax.set_yticklabels(feature_names[sorted_idx], fontsize=14)


        # Adjust layout
        plt.tight_layout()

        # Save the plot with high resolution
        plt.savefig('./results/xgboost_feature_importance.png', dpi=300)


    # Visualize the tree for Decision Tree
    if isinstance(model, DecisionTreeRegressor):
        max_depth = 20
        dot_data = export_graphviz(model, filled=True, feature_names=df.columns[input_indices],
                                   class_names=df.columns[output_indices], max_depth=max_depth)

        dot_file_path = f'./results/{model_name}_decision_tree.dot'
        with open(dot_file_path, 'w') as dot_file:
            dot_file.write(dot_data)

        png_file_path = f'./results/{model_name}_decision_tree.png'
        subprocess.run(['dot', '-Tpng', dot_file_path, '-o', png_file_path], check=True)

        max_depth = 4
        dot_data = export_graphviz(model, filled=True, feature_names=df.columns[input_indices],
                                   class_names=df.columns[output_indices], max_depth=max_depth)

        dot_file_path = f'./results/{model_name}_decision_tree_part.dot'
        with open(dot_file_path, 'w') as dot_file:
            dot_file.write(dot_data)

        png_file_path = f'./results/{model_name}_decision_tree_part.png'
        subprocess.run(['dot', '-Tpng', dot_file_path, '-o', png_file_path], check=True)

    if isinstance(model, KNeighborsRegressor):
        # Calculate permutation feature importance
        permutation_importance_results = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=RANDOM_SEED)

        # Print feature importances
        feature_importances = permutation_importance_results.importances_mean

        # Plot the feature importances
        sorted_idx = np.argsort(feature_importances)
        feature_names = np.array(input_feature_names)

        # Create a larger figure with higher resolution
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  # 10x8 inches figure with 300 dpi resolution

        # Plot feature importances
        bars = ax.barh(range(x_test.shape[1]), feature_importances[sorted_idx], color='red', height=0.5)

        # Set the title and increase the font size of y and x labels
        plt.title('Permutation Feature Importance', fontsize=16)
        ax.set_ylabel('Features', fontsize=14)
        ax.set_xlabel('Score', fontsize=14)
        # Increase the font size of y-tick labels and set feature names
        ax.set_yticks(np.arange(x_test.shape[1]))
        ax.set_yticklabels(feature_names[sorted_idx], fontsize=14)


        # Adjust layout
        plt.tight_layout()

        # Save the plot with high resolution
        plt.savefig('./results/permutation_feature_importance.png', dpi=300)

    # Make predictions on the training set and calculate metrics
    train_predictions = model.predict(x_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)

    # Make predictions on the test set and calculate metrics
    test_predictions = model.predict(x_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_r2_per_output = [r2_score(y_test[:, i], test_predictions[:, i]) for i in range(y_test.shape[1])]
    
    # Make predictions with 5-fold cross-validation
    cross_val_score_mse = -cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cross_val_score_mae = -cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    cross_val_score_mse_mean = cross_val_score_mse.mean()
    cross_val_score_mse_std = cross_val_score_mse.std()
    cross_val_score_mae_mean = cross_val_score_mae.mean()
    cross_val_score_mae_std = cross_val_score_mae.std()
        
    # Calculate MSE and MAE per output
    mse_per_output = [mean_squared_error(y_test[:, i], test_predictions[:, i]) for i in range(y_test.shape[1])]
    mae_per_output = [mean_absolute_error(y_test[:, i], test_predictions[:, i]) for i in range(y_test.shape[1])]
 

    # Save the results to a dictionary
    results[model_name] = {
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_r2_per_output': test_r2_per_output,
        'mse_per_output': mse_per_output,
        'mae_per_output': mae_per_output,
        'cross_val_score_mse_mean': cross_val_score_mse_mean,
        'cross_val_score_mse_std': cross_val_score_mse_std,
        'cross_val_score_mae_mean': cross_val_score_mae_mean,
        'cross_val_score_mae_std': cross_val_score_mae_std
    }
    
    # Save the results locally
    results_file_path = './results/results.json'
    with open(results_file_path, 'w') as results_file:
        json.dump(results, results_file, indent=4)

    # Make predictions on the entire dataset
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    all_predictions = model.predict(features)

    # Visualize the results
    grid_size = (5, 1)
    fig, axes = plt.subplots(*grid_size, figsize=(15, 20), dpi=100)
    axes = axes.flatten()

    for i, ax in enumerate(axes[:len(y_test[0])]):
        x_axis = range(1, len(target) + 1)
        ax.scatter(x_axis, target[:, i], color='black', label='Actual', s=100)
        ax.scatter(x_axis, all_predictions[:, i], color='red', label='Predicted', s=100)

        # Set the x-ticks to show every 2 instances
        step = 2
        x_ticks = list(range(1, len(target) + 1, step))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=20, rotation=45)  # Rotation and fontsize for x-tick labels
        
        #ax.set_title(target_labels[i], fontsize=24, pad=10)
        
        ax.set_xlabel('Batch', fontsize=18)
        ax.set_ylabel(target_labels[i], fontsize=20)
        ax.legend(fontsize=16)
        ax.tick_params(axis='x', rotation=0, labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        if 'temperature' in target_labels[i].lower():
            ax.set_ylabel(target_labels[i]+' (°C)', fontsize=18)
        if "duration" in target_labels[i].lower():
            ax.set_ylabel(target_labels[i]+' (days)', fontsize=18)
        if "compost" in target_labels[i].lower():
            ax.set_ylabel(target_labels[i]+' (kg)', fontsize=18)

    plt.suptitle(model_names[num], fontsize=22)
    plt.tight_layout(pad=3.0)
    plt.savefig(f'./results/{model_names[num]}_plot.png', dpi=100)