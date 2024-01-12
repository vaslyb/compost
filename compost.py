import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb
import os

# Create a results directory if it does not exist
results_directory = './results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    
# Load CSV file using Pandas
df = pd.read_csv('compost.csv')
df = df.replace(',', '.', regex=True)

input = [1,2,3,4,5,6]
output = [7,8,9,10,11]

input_feature_names = df.columns[input]
output_feature_names = df.columns[output]

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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test_temp = x_test
x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)

# Define regression models
models = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=41),
    MultiOutputRegressor(LinearSVR()),
    XGBRegressor()
]

model_names = ['Linear Regression', 'K-Neighbors Regressor', 'Decision Tree Regressor', 'Support Vector Regressor', 'XGBoost Regressor']

results = {}

predictions_df = pd.DataFrame()

for model in models:
    model_name = type(model).__name__
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Visualize feature importance for XGBoost
    if isinstance(model, XGBRegressor):
        # Visualize feature importance for XGBoost
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax, importance_type='weight', show_values=False)
        ax.set_yticklabels(['Biowaste Feed', 'Pruning Feed', 'Recycled Compost Feed', 'Sawdust Feed', 'Leaf Feed', 'Ambient Temperature'])
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('./results/XGBoost_feature_importance.png')
    
    # Visualize the tree for Decision Tree
    if isinstance(model, DecisionTreeRegressor):
        max_depth = 20 
        
        dot_data = export_graphviz(model, filled=True, feature_names=df.columns[input],
                                    class_names=df.columns[output], max_depth=max_depth)
        
        dot_file_path = f'./results/{model_name}_decision_tree.dot'
        with open(dot_file_path, 'w') as dot_file:
            dot_file.write(dot_data)
        
        png_file_path = f'./results/{model_name}_decision_tree.png'
        subprocess.run(['dot', '-Tpng', dot_file_path, '-o', png_file_path], check=True)
   
    
    # Visualize part of the tree for Decision Tree     
    if isinstance(model, DecisionTreeRegressor):
        max_depth = 4  

        dot_data = export_graphviz(model, filled=True, feature_names=df.columns[input],
                                    class_names=df.columns[output], max_depth=max_depth)
        
        dot_file_path = f'./results/{model_name}_decision_tree_part.dot'
        with open(dot_file_path, 'w') as dot_file:
            dot_file.write(dot_data)
        
        png_file_path = f'./results/{model_name}_decision_tree_part.png'
        subprocess.run(['dot', '-Tpng', dot_file_path, '-o', png_file_path], check=True)
           
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
    
    # Make the predictions on the entire dataset
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    all_predictions = model.predict(features)
    
    # Visualize the results
    if isinstance(model, KNeighborsRegressor):
        # Visualize the results
        grid_size = (5, 1)

        fig, axes = plt.subplots(*grid_size, figsize=(15, 15))

        axes = axes.flatten()
        for i, ax in enumerate(axes[:len(y_test[0])]):
            x_axis = [j for j in range(1, len(target) + 1)]  
            ax.scatter(x_axis, target[:, i], color='black', label='Actual')

            ax.scatter(x_axis, all_predictions[:, i], color='red', label='Predicted')
            ax.set_xlabel('Run')
            ax.set_ylabel(target_labels[i])
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('K-Nearest Neighbors Regressor')
        plt.tight_layout(pad=3.0)
        plt.savefig('./results/model_plot.png')

    for i, output_feature_name in enumerate(output_feature_names):
        ground_truth_col = f'Ground Truth_{model_name}_{output_feature_name}'
        predictions_col = f'Predictions_{model_name}_{output_feature_name}'

        ground_truth_values = y_test[:, i]
        predictions_values = test_predictions[:, i]

        output_feature_predictions_df = pd.DataFrame({
            ground_truth_col: ground_truth_values,
            predictions_col: predictions_values,
        })

        predictions_df = pd.concat([predictions_df, output_feature_predictions_df], axis=1)

    for j, input_feature_name in enumerate(input_feature_names):
        input_feature_col = f'Input_{input_feature_name}'
        input_feature_values = x_test_temp[:, j]

        input_feature_df = pd.DataFrame({
            input_feature_col: input_feature_values,
        })

        predictions_df = pd.concat([predictions_df, input_feature_df], axis=1)

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
    
# Save the predictions to a CSV file
predictions_df.to_csv('./results/all_models_predictions.csv', index=False)

# Save results to a JSON file
with open('./results/compost_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4, default=lambda x: x.tolist())

# Create a table with the performance of each model 
metrics_table_data = []
for model_name, result in zip(model_names, results.values()):
    metrics_table_data.append([
        model_name,
        result['test_mse'],
        result['test_mae'],
        result['test_r2']
    ])

metrics_table_df = pd.DataFrame(metrics_table_data, columns=[
    'Model', 'MSE', 'MAE', 'R2'
])

metrics_table_df = metrics_table_df.round(3)

metrics_table_df = metrics_table_df.sort_values(by='R2', ascending=True)

best_model_name = metrics_table_df.iloc[-1]['Model']

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
table = ax.table(cellText=metrics_table_df.round(3).values,
                 colLabels=metrics_table_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1],  
                 cellColours=[['#f0f0f0'] * len(metrics_table_df.columns)] * len(metrics_table_df),
                 colColours=['#ffffff'] * len(metrics_table_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(17)
table.auto_set_column_width([0, 1, 2, 3])  
table.auto_set_font_size(False)
table.set_fontsize(17)
table.auto_set_column_width([0, 1, 2, 3])
table.auto_set_column_width(1)

best_model_index = metrics_table_df.index[metrics_table_df['Model'] == best_model_name][0] + 1
for coord, cell in table._cells.items():
    row, col = coord
    cell_obj = table._cells[row, col]
    if row == len(metrics_table_df):
        cell_obj.set_text_props(fontweight='bold')

csv_file_path = './results/metrics_table.csv'
metrics_table_df.to_csv(csv_file_path, index=False)
plt.title('Metrics Table', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./results/metrics_table.png')

# Plot the coefficients as a heatmap
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

