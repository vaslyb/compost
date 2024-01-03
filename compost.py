import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_standardized, target_standardized, test_size=0.15, random_state=42)

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
    
    # Make predictions on the test set
    test_predictions = model.predict(x_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    # Store results in the dictionary
    results[model_name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'coefficients': list(model.coef_) if hasattr(model, 'coef_') else None
    }

# Print and save results to a JSON file
print("Results:")
for model_name, result in results.items():
    print(f"{model_name}: Train MSE = {result['train_mse']}, Test MSE = {result['test_mse']}")
    print(f"{model_name}: Train MAE = {result['train_mae']}, Test MAE = {result['test_mae']}")

# Save results to a JSON file
with open('compost_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4, default=lambda x: x.tolist())

print("Results saved to 'compost_results.json'")

# Plot coefficients using Plotly
for model_name, result in results.items():
    coefficients = result['coefficients']
    if coefficients is not None:
        fig = go.Figure(data=go.Heatmap(
            z=coefficients * 100,
            x=df.columns[1:7],
            y=df.columns[-5:],
            colorscale='YlOrRd',
            colorbar=dict(title='Coefficients')
        ))
        fig.update_layout(title=f'{model_name} Coefficients Heatmap', xaxis_title='Features', yaxis_title='Target Variables')

        # Save the figure as an image file (e.g., PNG)
        fig.write_image(f'{model_name}_heatmap.png')

