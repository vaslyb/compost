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

df = df.replace(',', '.', regex=True)  

# Extract features and target from the DataFrame
features = df.iloc[:, [7,8,9,10,11]].values  
target = df.iloc[:, [13,14,17,18] ].values 
#[8,9,10,11,12]
#[13,14,17,18] 
#[12,13,14,15,16] 
selected_indices = [5, 10, 15, 20, 25 , 30, 35, 40]
# Create a boolean mask for the selected indices
mask = np.zeros(features.shape[0], dtype=bool)
mask[selected_indices] = True

# Use boolean indexing to get the selected and remaining rows
x_train = features[mask]
x_test = features[~mask]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

target = target.astype(float)
y_train = target[mask]
y_test = target[~mask]
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

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
        plot_tree(model, filled=True, feature_names=df.columns[[7,8,9,10,11]], class_names=df.columns[[13,14,17,18] ], fontsize=10, max_depth=max_depth)
        plt.title(f'{model_name} Decision Tree (Max Depth = {max_depth})')
        image_file_path = f'./results/{model_name}_decision_tree.png'
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

    # Save predictions and ground truth to a CSV file
    predictions_df = pd.DataFrame({
        'Ground Truth': y_test.flatten(),
        'Predictions': test_predictions.flatten()
    })

    predictions_df.to_csv(f'./results/{model_name}_predictions.csv', index=False)
    print(f'Test predictions saved to {model_name}_predictions.csv')
    if hasattr(model, 'intercept_'):
        print(model.intercept_)
    # Store results in the dictionary
    results[model_name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coefficients': list(model.coef_) if hasattr(model, 'coef_') else None,
        'predictions_file': f'{model_name}_predictions.csv'
    }

# Print and save results to a JSON file
print("Results:")
for model_name, result in results.items():
    print(f"{model_name}: Train MSE = {result['train_mse']}, Test MSE = {result['test_mse']}")
    print(f"{model_name}: Train MAE = {result['train_mae']}, Test MAE = {result['test_mae']}")
    print(f"{model_name}: Train R2 = {result['train_r2']}, Test R2 = {result['test_r2']}")

# Save results to a JSON file
with open('./results/compost_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4, default=lambda x: x.tolist())

print("Results saved to 'compost_results.json'")
print()
# Plot coefficients using Plotly
for model_name, result in results.items():
    coefficients = result['coefficients']
    if coefficients is not None:
        fig = go.Figure(data=go.Heatmap(
            z=coefficients,
            x=df.columns[[7,8,9,10,11]],
            y=df.columns[[13,14,17,18] ],
            colorscale='YlOrRd',
            colorbar=dict(title='Coefficients')
        ))

        # Add annotations with coefficient values
        for i in range(len(df.columns[[13,14,17,18] ])):
            for j in range(len(df.columns[[7,8,9,10,11]])):
                coefficient_value = float(coefficients[i][j])
                fig.add_annotation(
                    x=df.columns[[7,8,9,10,11]][j],
                    y=df.columns[[13,14,17,18]][i],
                    text=f'{coefficient_value:.2f}',
                    showarrow=False,
                    font=dict(color='black', size=10)
                )

        fig.update_layout(title=f'{model_name} Coefficients Heatmap', xaxis_title='Features', yaxis_title='Target Variables')

        # Save the figure as an image file (e.g., PNG)
        fig.write_image(f'./results/{model_name}_heatmap.png')

