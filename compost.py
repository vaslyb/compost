import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb

# Load CSV file using Pandas
df = pd.read_csv('compost.csv') 
df = df.replace(',', '.', regex=True)  

# Select input and output of the models
# 3 different experiments

# feed-duration
# input = [1]
# output = [14]

# temperatures
# input = [11]
# output = [12,13]

# percentage of feed
# input = [7,8,9,10]
# output = [12,13,17,18]

# all consistancy - feed
# input = [2,3,4,5,6,11]
# output = [12,13,14,15,16]

# all consistancy - feed
input = [2,3,4,5,6,11]
output = [12,13,14,15,16]

features = df.iloc[:, input].values  
target = df.iloc[:, output].values
target_labels = df.columns[output] 

# Collect the test set in order to have representative data
selected_indices = [5, 10, 15, 20, 25 , 30, 35, 40]
mask = np.zeros(features.shape[0], dtype=bool)
mask[selected_indices] = True

x_train = features[~mask]
x_test = features[mask]

target = target.astype(float)
y_train = target[~mask]
y_test = target[mask]

# Standarise the input data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Standarise the output data
# scaler = StandardScaler()
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)

# Define regression models
models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    MultiOutputRegressor(LinearSVR()),
    XGBRegressor()
]

flag = True

results = {}

for model in models:
    model_name = type(model).__name__
    
    # Train the model
    model.fit(x_train, y_train)
    
    if isinstance(model, XGBRegressor):
        # Visualize feature importance for XGBoost
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax, importance_type='weight', show_values=False)
        ax.set_yticklabels(['Biowaste Feed', 'Pruning Feed', 'Recycled Compost Feed', 'Sawdust Feed', 'Leaf Feed', 'Ambient Temperature'])
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('./results/XGBoost_feature_importance.png')

        # # Visualize individual trees in the XGBoost model
        # for tree_index in range(model.n_estimators):
        #     plt.figure(figsize=(10, 8))
        #     xgb.plot_tree(model, num_trees=tree_index, rankdir='LR', feature_names=['Biowaste Feed', 'Pruning Feed', 'Recycled Compost Feed', 'Sawdust Feed', 'Leaf Feed', 'Ambient Temperature'])
        #     plt.title(f'XGBoost Tree {tree_index + 1}')
        #     plt.savefig(f'./results/XGBoost_tree_{tree_index + 1}.png')
        #     plt.show()
    
    # Visualize the tree for Decision Tree
    if isinstance(model, DecisionTreeRegressor):
        # Visualize a limited-depth decision tree
        max_depth = 4  # Set the maximum depth you want to visualize
        
        # Create a DOT format string for the decision tree
        dot_data = export_graphviz(model, filled=True, feature_names=df.columns[input],
                                    class_names=df.columns[output], max_depth=max_depth)
        
        # Save the DOT string to a file
        dot_file_path = f'./results/{model_name}_decision_tree.dot'
        with open(dot_file_path, 'w') as dot_file:
            dot_file.write(dot_data)
        
        png_file_path = f'./results/{model_name}_decision_tree.png'
        subprocess.run(['dot', '-Tpng', dot_file_path, '-o', png_file_path], check=True)
   
        print(f'Decision tree visualization saved to {png_file_path}')

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

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    all_predictions = model.predict(features)
    
    if isinstance(model, KNeighborsRegressor) and flag==True:
        # Visualize the results
        grid_size = (5, 1)

        fig, axes = plt.subplots(*grid_size, figsize=(15, 15))

        axes = axes.flatten()
        for i, ax in enumerate(axes[:len(y_test[0])]):
            x_axis = [j for j in range(1, len(target) + 1)]  # Update x_axis for each subplot

            # Scatter plot for actual values
            ax.scatter(x_axis, target[:, i], color='black', label='Actual')

            # Scatter plot for predicted values
            ax.scatter(x_axis, all_predictions[:, i], color='red', label='Predicted')
            ax.set_xlabel('Run')
            ax.set_ylabel(target_labels[i])
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('K-Neighbors Regressor')
        # Adjust layout and save the figure
        plt.tight_layout(pad=3.0)
        plt.savefig('./results/svr_plot.png')

    # Save predictions and ground truth to a CSV file
    predictions_df = pd.DataFrame({
        'Ground Truth': y_test.flatten(),
        'Predictions': test_predictions.flatten()
    })

    predictions_df.to_csv(f'./results/{model_name}_predictions.csv', index=False)
    print(f'Test predictions saved to {model_name}_predictions.csv')
    if hasattr(model, 'intercept_'):
        print("Linear's Coefficients",model.coef_,model.intercept_)
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
            x=df.columns[input],
            y=df.columns[output ],
            colorscale='YlOrRd',
            colorbar=dict(title='Coefficients')
        ))

        # Add annotations with coefficient values
        for i in range(len(df.columns[output])):
            for j in range(len(df.columns[input])):
                coefficient_value = float(coefficients[i][j])
                fig.add_annotation(
                    x=df.columns[input][j],
                    y=df.columns[output][i],
                    text=f'{coefficient_value:.2f}',
                    showarrow=False,
                    font=dict(color='black', size=10)
                )

        fig.update_layout(title=f'{model_name} Coefficients Heatmap', xaxis_title='Features', yaxis_title='Target Variables')

        # Save the figure as an image file (e.g., PNG)
        fig.write_image(f'./results/{model_name}_heatmap.png')

