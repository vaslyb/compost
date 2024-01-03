import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
import pandas as pd

# Load CSV file using Pandas
df = pd.read_csv('compost.csv')

gi
# Load the CSV file
data = np.loadtxt('compost.csv', delimiter=',', skiprows=1)  
header = np.loadtxt('compost.csv', delimiter=',', max_rows=1)

# Keep labels
target_column_index = -1

# Extract features 
features = data[:, :target_column_index]

# Extract the target column
target = data[:, target_column_index]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

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
    
    # Make predictions on the training set
    train_predictions = model.predict(x_train)
    train_score = mean_squared_error(y_train, train_predictions)
    
    # Make predictions on the test set
    test_predictions = model.predict(x_test)
    test_score = mean_squared_error(y_test, test_predictions)
    
    # Store results in the dictionary
    results[model_name] = {
        'train_score': train_score,
        'test_score': test_score,
        'coefficients': list(model.coef_) if hasattr(model, 'coef_') else None
    }

# Print and save results to a JSON file
print("Results:")
for model_name, result in results.items():
    print(f"{model_name}: Train Score = {result['train_score']}, Test Score = {result['test_score']}")

# Save results to a JSON file
with open('compost_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to 'compost_results.json'")