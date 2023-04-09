import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import warnings
pd.options.mode.chained_assignment = None  # default='warn
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def process_training_data(raw_data):
    raw_data['next_period_profit'] = (raw_data['output_X'] * (raw_data['output_own_price'] - raw_data['output_own_cost'])).shift(-1)
    processed_data = raw_data[['output_X', 'output_own_price', 'output_own_cost', 'output_comp_price', 'output_own_share']]
    processed_data = raw_data[['output_date', 'output_X', 'output_own_price', 'output_own_cost', 'output_comp_price', 'output_own_share', 'output_own_sales', 'next_period_profit']]
    processed_data.dropna(inplace=True)
    processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return processed_data

def train_model(processed_data, use_hyperparameters, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf):
    X = processed_data.drop(['output_date', 'output_own_price', 'next_period_profit'], axis=1)
    y = processed_data['next_period_profit']
    #the split raito is 20:80
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_hyperparameters=="True":
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
    else:
        model = GradientBoostingRegressor().fit(X_train, y_train)
    
    return model


def predict_price(predicted_market_conditions, predicted_per_unit_cost, past_data, model):
    market_data = past_data[past_data['mkt_id'] == predicted_market_conditions['mkt_id']]
    #past_prices = market_data['output_own_price'].tail(3)
    past_sales = market_data['output_own_sales'].tail(3)
    past_share = market_data['output_own_share'].tail(3)
    average_comp_price = market_data['output_comp_price'].tail(1).values[0]
    market_condition = predicted_market_conditions['output_X']
    inputs = np.array([market_condition, predicted_per_unit_cost, average_comp_price, past_share.mean(), past_sales.mean()])
    predicted_profit = model.predict(inputs.reshape(1, -1))[0]
    optimal_price = predicted_per_unit_cost + (predicted_profit / predicted_market_conditions['output_own_sales'])
    return optimal_price


# Load sample data
raw_data = pd.read_csv('output_data.csv')

# Process training data
print("the training data is started...")
processed_data = process_training_data(raw_data)

# Hyperparameters can be changes by the user. This values are very optimal
n_estimators = 200
learning_rate = 0.05
max_depth = 8
min_samples_split = 500
min_samples_leaf = 50

"""
    The user can activate or deactivate the hyperparameterization
        no need to write False, anything but not True is fine
                                                                """

flag="True"

model = train_model(processed_data,flag, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf)
print("the model trained successfully")
# Predict the optimal price for the next period
predicted_market_conditions = {'mkt_id': 164, 'output_X':37.23, 'output_own_sales':125}
predicted_per_unit_cost = 7
print("the past data are filtering....")
#the past data are filtering based on the date
past_data = raw_data[pd.to_datetime(raw_data['output_date']) < pd.to_datetime('12jan2019')]
optimal_price = predict_price(predicted_market_conditions, predicted_per_unit_cost, past_data, model)

print('The recommended price for the next period is: $',optimal_price)
