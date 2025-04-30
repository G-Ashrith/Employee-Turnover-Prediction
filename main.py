import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Configure Seaborn plot styles: Set background color and use dark grid
sns.set(rc={'axes.facecolor': '#f1faeb'}, style='darkgrid')
# Load dataset
df = pd.read_csv('f:/Downloads/XGBoost-CatBoost-Employee-Resignation-master/XGBoost-CatBoost-Employee-Resignation-master/HR.csv')
print(df.head())
# Display a concise summary of the dataframe
print(df.info())
# Define the continuous features
continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']

# Identify the features to be converted to object data type
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]

# Convert the identified features to object data type
df[features_to_convert] = df[features_to_convert].astype('object')

print(df.dtypes)
# Get the summary statistics for numerical variables
df.describe().T
# Get the summary statistics for categorical variables
df.describe(include='object')
# Create subplots for kde plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for ax, col in zip(axes.flatten(), continuous_features):
    sns.kdeplot(data=df, x=col, fill=True, linewidth=2, hue='left', ax=ax, palette = {0: '#009c05', 1: 'darkorange'})
    ax.set_title(f'{col} vs Target')

axes[2,1].axis('off')
plt.suptitle('Distribution of Continuous Features by Target', fontsize=22)
plt.tight_layout()
plt.show()
# List of categorical features
cat_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Plot each feature
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x=cat_features[i], hue='left', data=df, ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(cat_features[i])
    ax.set_ylabel('Count')
    ax.set_xlabel('')
    ax.legend(title='Left', loc='upper right')

plt.suptitle('Distribution of Categorical Features by Target', fontsize=22)
plt.tight_layout()
plt.show()
# Reload the dataset to retain the original data types of the variables
df = pd.read_csv('f:/Downloads/XGBoost-CatBoost-Employee-Resignation-master/XGBoost-CatBoost-Employee-Resignation-master/HR.csv')
# Generate the missing values matrix using missingno.bar()
msno.bar(df, color='#009c05')

# Display the plot
plt.show()

# Implementing one-hot encoding on the 'sales' feature
df_encoded = pd.get_dummies(df, columns=['sales'], drop_first=True)

# Label encoding of 'salary' feature
le = LabelEncoder()
df_encoded['salary'] = le.fit_transform(df_encoded['salary'])

df_encoded.head()

df_encoded.dtypes

# Define the features (X) and the output labels (y)
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Define the model
xgb_base = xgb.XGBRegressor(objective ='reg:squarederror')

def tune_regressor_hyperparameters(reg, param_grid, X_train, y_train, scoring='neg_mean_squared_error', n_splits=3):
    '''
    This function optimizes the hyperparameters for a regressor by searching over a specified hyperparameter grid. 
    It uses GridSearchCV and cross-validation (KFold) to evaluate different combinations of hyperparameters. 
    The combination with the highest negative mean squared error is selected as the default scoring metric. 
    The function returns the regressor with the optimal hyperparameters.
    '''
    
    # Create the cross-validation object using KFold
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the GridSearchCV object
    reg_grid = GridSearchCV(reg, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    reg_grid.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyperparameters = reg_grid.best_params_
    
    # Return best_estimator_ attribute which gives us the best model that has been fitted to the training data
    return reg_grid.best_estimator_, best_hyperparameters

# Define the parameters for grid search
xgb_param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [200, 250, 300],
    'min_child_weight': [2, 3, 4]
}
# Tune the hyperparameters
best_xgb, best_xgb_hyperparameters = tune_regressor_hyperparameters(xgb_base, xgb_param_grid, X_train, y_train)

print('XGBoost Regressor Optimal Hyperparameters: \n', best_xgb_hyperparameters)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for training data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)

    # Calculate metrics for testing data
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(data = [mae_test, mse_test, rmse_test, r2_test],
                              index = ['MAE', 'MSE', 'RMSE', 'R2 Score'],
                              columns = [model_name])
    
    # Print the metrics
    print(f"{model_name} Training Data Metrics:")
    print("MAE: {:.4f}".format(mae_train))
    print("MSE: {:.4f}".format(mse_train))
    print("RMSE: {:.4f}".format(rmse_train))
    print("R2 Score: {:.4f}".format(r2_train))
    
    print(f"\n{model_name} Testing Data Metrics:")
    print("MAE: {:.4f}".format(mae_test))
    print("MSE: {:.4f}".format(mse_test))
    print("RMSE: {:.4f}".format(rmse_test))
    print("R2 Score: {:.4f}".format(r2_test))
        
    return metrics_df

xgb_result = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, 'XGBoost')

# Define the model
ctb_base = CatBoostRegressor(verbose=0)

# Define the parameters for grid search
ctb_param_grid = {
    'iterations': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
}

# Tune the hyperparameters
best_ctb, best_ctb_hyperparameters = tune_regressor_hyperparameters(ctb_base, ctb_param_grid, X_train, y_train)

print('\nCatBoost Regressor Optimal Hyperparameters: \n', best_ctb_hyperparameters)

ctb_result = evaluate_model(best_ctb, X_train, y_train, X_test, y_test, 'CatBoost')

# Combine the dataframes
combined_df = pd.concat([ctb_result.T, xgb_result.T], axis=0)
combined_df['Model'] = ['CatBoost', 'XGBoost']

# Melt the dataframe
melted_df = combined_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Define custom colors
custom_colors = ['#009c05', 'darkorange']

# Create the barplot
plt.figure(figsize=(10,6))
sns.barplot(x='Score', y='Metric', hue='Model', data=melted_df, palette=custom_colors)

plt.title('Model Comparison')
plt.show()

print(combined_df)

