import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

# Load CSV file using Pandas
df = pd.read_csv('compost.csv') 

df = df.replace(',', '.', regex=True)  # Replace commas with dots (assuming ',' represents decimal separator)

# Extract features and target from the DataFrame
features = df.iloc[:, 1:7].values  
target = df.iloc[:, -5:].values  

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

target = target.astype(float)
target_standardized = scaler.fit_transform(target)

# Select specific rows from the array with indices 1, 2, 3
selected_indices = [5, 10, 15, 20, 25 , 30, 35, 40]

# Create a boolean mask for the selected indices
mask = np.zeros(features_standardized.shape[0], dtype=bool)
mask[selected_indices] = True

# Use boolean indexing to get the selected and remaining rows
x_train = features_standardized[mask]
y_train = target_standardized[mask]

x_test = features_standardized[~mask]
y_test = target_standardized[~mask]

# Define a list of regression models
models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    MultiOutputRegressor(LinearSVR())
]

# Dictionary to store results
results = {}

# Loop through each model
for model in models:
    model_name = type(model).__name__
    
    # Train the model
    model.fit(x_train, y_train)
    if isinstance(model, DecisionTreeRegressor):
        # Visualize a limited-depth decision tree
        max_depth = 3  # Set the maximum depth you want to visualize
        plt.figure(figsize=(18, 12))
        plot_tree(model, filled=True, feature_names=df.columns[1:7], class_names=df.columns[-5:], fontsize=10, max_depth=max_depth)
        plt.title(f'{model_name} Decision Tree (Max Depth = {max_depth})')
        image_file_path = f'{model_name}_decision_tree.png'
        plt.savefig(image_file_path)
        print(f'Decision tree visualization saved to {image_file_path}')

    
    # Make predictions on the training set
    train_predictions = model.predict(x_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    
    # Make predictions on the test set
    test_predictions = model.predict(x_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # Store results in the dictionary
    results[model_name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coefficients': list(model.coef_) if hasattr(model, 'coef_') else None
    }

# Print and save results to a JSON file
print("Results:")
for model_name, result in results.items():
    print(f"{model_name}: Train MSE = {result['train_mse']}, Test MSE = {result['test_mse']}")
    print(f"{model_name}: Train MAE = {result['train_mae']}, Test MAE = {result['test_mae']}")
    print(f"{model_name}: Train R2 = {result['train_r2']}, Test R2 = {result['test_r2']}")

# Save results to a JSON file
with open('compost_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4, default=lambda x: x.tolist())

print("Results saved to 'compost_results.json'")
print()
# Plot coefficients using Plotly
for model_name, result in results.items():
    coefficients = result['coefficients']
    if coefficients is not None:
        fig = go.Figure(data=go.Heatmap(
            z=coefficients,
            x=df.columns[1:7],
            y=df.columns[-5:],
            colorscale='YlOrRd',
            colorbar=dict(title='Coefficients')
        ))

        # Add annotations with coefficient values
        for i in range(len(df.columns[-5:])):
            for j in range(len(df.columns[1:7])):
                coefficient_value = float(coefficients[i][j])
                fig.add_annotation(
                    x=df.columns[1:7][j],
                    y=df.columns[-5:][i],
                    text=f'{coefficient_value:.2f}',
                    showarrow=False,
                    font=dict(color='black', size=10)
                )

        fig.update_layout(title=f'{model_name} Coefficients Heatmap', xaxis_title='Features', yaxis_title='Target Variables')

        # Save the figure as an image file (e.g., PNG)
        fig.write_image(f'{model_name}_heatmap.png')

