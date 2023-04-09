# Price-Modeling-GradientBoosting
-------------------------the fields should be customized based on your dataset's fields-------------------------

This is a pricing optimization model using Gradient Boosting Regression. It predicts the optimal price based on market conditions and past sales data. Users can adjust hyperparameters and values based on their needs. It improves profitability and competitiveness.


Python script for training a Gradient Boosting Regression model to predict the
optimal price for a product in a given market
The script reads data from a CSV file ('output_data.csv'), processes it by selecting relevant
columns and removing rows with missing or infinite values, and then trains a
GradientBoostingRegressor model on this data.

next_period_profit = output_X *( output_own_price - output_own_cost)

The user can choose whether or not to use hyperparameters for the model by specifying the
flag input (flag="True" for using hyperparameters, anything but not True not for using
hyperparameters). If hyperparameters are used, the user can also change the values of the
hyperparameters n_estimators, learning_rate, max_depth, min_samples_split, and
min_samples_leaf by modifying their values in the script.

● n_estimators: This is the number of decision trees that will be used in the boosting
process. Increasing the number of trees can increase the model's performance, but
may also increase the risk of overfitting the training data.

● learning_rate: This is the amount that each decision tree is allowed to correct the
errors of the previous tree. A lower learning rate can lead to a more conservative
model that generalizes better to new data, while a higher learning rate can lead to a
more complex model that is better at fitting the training data.
● max_depth: This is the maximum depth of each decision tree. Increasing the depth
can increase the model's ability to fit complex data, but may also increase the risk of
overfitting the training data.

● min_samples_split: This is the minimum number of samples required to split an
internal node in a decision tree. Increasing this value can lead to simpler models that
are less likely to overfit, but may also result in a model that is less expressive and
thus less accurate.

● min_samples_leaf: This is the minimum number of samples required to be at a leaf
node in a decision tree. Similar to min_samples_split, increasing this value can lead
to simpler models that are less likely to overfit, but may also result in a less
expressive model.

The effect of these hyperparameters on the model's performance can vary depending on the
specific dataset and problem being addressed. In general, finding the optimal combination of
hyperparameters requires some trial and error and experimentation with different values.
ps: The default values are very optimistic for this caase(nearlly 20000 rows of data for 55
different id), but still, you can change the values

Once the model is trained, the user can use the function 'predict_price' to predict the
optimal price for a given market condition. The function takes as input a dictionary of
predicted market conditions, including the market ID, output_X (market condition), and
output_own_sales (product sales), as well as a predicted per-unit cost and a data frame of
past data. The function then uses the trained model to predict the optimal price for the next
period based on the input data. Finally, the script prints out the recommended price for the
next period based on the inputs provided to the 'predict_price' function


Description of variables in data:

• “mkt_id” - identifier for the market

• “output_date” - identifier for the period (day)

• “output_own_price” - own price set in the period (day)

• “output_own_cost” - own per-unit cost of goods sold for the period

• “output_comp_price” - average of competitor prices in the period

• “output_X” - a variable summarizing market conditions in the period (on a scale between 0 and 100)

• “output_own_sales” - own sales in the period

• “Output_own_share” - own sales share in the period

• “Output_own_profits” - own total profits in the period

