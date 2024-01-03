import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load CSV file using Pandas
df = pd.read_csv('compost.csv', skiprows=1) 

df = df.replace(',', '.', regex=True)  # Replace commas with dots (assuming ',' represents decimal separator)

# Extract features and target from the DataFrame
features = df.iloc[:, 1:5].values  
target = df.iloc[:, -4:].values  

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
    json.dump(results, json_file, indent=4, default=lambda x: x.tolist())

print("Results saved to 'compost_results.json'")