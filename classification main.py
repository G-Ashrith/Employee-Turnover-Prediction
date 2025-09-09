import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Configure Seaborn plot styles
sns.set(rc={'axes.facecolor': '#f1faeb'}, style='darkgrid')

# Load dataset
df = pd.read_csv('HR.csv')
print(df.head())
print(df.info())

# Continuous features
continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project',
                       'average_montly_hours', 'time_spend_company']

# Convert non-continuous to object
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]
df[features_to_convert] = df[features_to_convert].astype('object')

print(df.dtypes)

# EDA: distributions
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
for ax, col in zip(axes.flatten(), continuous_features):
    sns.kdeplot(data=df, x=col, fill=True, linewidth=2, hue='left',
                ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(f'{col} vs Target')
axes[2,1].axis('off')
plt.suptitle('Distribution of Continuous Features by Target', fontsize=22)
plt.tight_layout()
plt.show()

# Categorical distributions
cat_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x=cat_features[i], hue='left', data=df, ax=ax,
                  palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(cat_features[i])
plt.suptitle('Distribution of Categorical Features by Target', fontsize=22)
plt.tight_layout()
plt.show()

# Missing values check
df = pd.read_csv('HR.csv')  # reload original datatypes
msno.bar(df, color='#009c05')
plt.show()

# Encoding
df_encoded = pd.get_dummies(df, columns=['sales'], drop_first=True)
le = LabelEncoder()
df_encoded['salary'] = le.fit_transform(df_encoded['salary'])

# Define X and y
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Generic tuner
def tune_classifier_hyperparameters(clf, param_grid, X_train, y_train, scoring='accuracy', n_splits=3):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    metrics_df = pd.DataFrame({
        model_name: [acc, prec, rec, f1, roc_auc]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'])

    print(f"\n{model_name} Classification Report")
    print("Accuracy:  {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall:    {:.4f}".format(rec))
    print("F1-score:  {:.4f}".format(f1))
    print("ROC-AUC:   {:.4f}".format(roc_auc))

    return metrics_df

# --------------------
# XGBoost
# --------------------
xgb_base = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                         use_label_encoder=False, random_state=0)

xgb_param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [50, 100],
    'min_child_weight': [2, 3]
}

best_xgb, best_xgb_params = tune_classifier_hyperparameters(xgb_base, xgb_param_grid, X_train, y_train)
print("\nBest XGBoost Params:", best_xgb_params)
xgb_result = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, 'XGBoost')

# --------------------
# CatBoost
# --------------------
ctb_base = CatBoostClassifier(verbose=0, random_state=0)

ctb_param_grid = {
    'iterations': [50, 100],
    'learning_rate': [0.1, 0.3],
    'depth': [4, 6],
    'l2_leaf_reg': [1, 3]
}

best_ctb, best_ctb_params = tune_classifier_hyperparameters(ctb_base, ctb_param_grid, X_train, y_train)
print("\nBest CatBoost Params:", best_ctb_params)
ctb_result = evaluate_model(best_ctb, X_train, y_train, X_test, y_test, 'CatBoost')

# --------------------
# Compare Results
# --------------------
combined_df = pd.concat([ctb_result.T, xgb_result.T], axis=0)
combined_df['Model'] = ['CatBoost', 'XGBoost']

# Melt for plotting
melted_df = combined_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Score', y='Metric', hue='Model', data=melted_df, palette=['#009c05', 'darkorange'])
plt.title('Model Comparison (Classification)')
plt.show()

print("\nFinal Comparison:\n", combined_df)
