import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import Counter

# Machine Learning libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- UI Configuration ---
st.set_page_config(
    page_title="Customizable ML Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Customizable Employee Turnover Prediction")
st.markdown("""
Upload your dataset, select the columns you want to use, and train a
machine learning model to predict employee turnover.
""")

# --- Data Loading Section ---
st.header("1. Upload and View Your Dataset")
st.markdown("For this app to work, please run it using `streamlit run <your-filename>.py` from your terminal.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File loaded successfully!")
    st.session_state['data'] = data
else:
    st.info("No file uploaded. Loading default `HR.csv` for demonstration.")
    data = pd.read_csv('HR.csv')
    st.session_state['data'] = data

st.subheader("Dataset Preview")
st.dataframe(st.session_state['data'].head())

# --- Feature and Target Selection ---
st.header("2. Select Features and Target")
all_columns = st.session_state['data'].columns.tolist()

# Allow the user to select the target variable
target_column = st.selectbox(
    "Select the target variable (the column you want to predict):",
    options=all_columns,
    index=all_columns.index('Attrition') if 'Attrition' in all_columns else (all_columns.index('left') if 'left' in all_columns else 0)
)

# Allow the user to select the features (input variables)
feature_columns = st.multiselect(
    "Select the features to use for training:",
    options=[col for col in all_columns if col != target_column],
    default=[col for col in all_columns if col != target_column and col != 'Employee_ID']
)

# --- Model Training and Evaluation ---
if st.button("Train Models", use_container_width=True):
    if not feature_columns:
        st.error("Please select at least one feature column to train the model.")
    else:
        # Use a caching decorator to prevent re-training on every interaction
        @st.cache_resource
        def train_and_evaluate_models(df, features, target):
            st.info("Training models... This may take a moment.")
            
            try:
                # Prepare data
                X = df[features]
                y = df[target]

                # --- NEW: Check and preprocess the target variable (y) ---
                if y.dtype == 'object':
                    st.info(f"Target variable '{target}' is categorical. Automatically converting to numerical labels.")
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y), index=y.index)
                
                # Check for class imbalance in the target variable
                counts = Counter(y)
                st.info(f"Class distribution of the target variable: {counts}")
                minority_class_count = min(counts.values())
                majority_class_count = max(counts.values())
                imbalance_ratio = majority_class_count / minority_class_count
                if imbalance_ratio > 2:
                    st.warning(f"Class imbalance detected! The ratio of the majority to minority class is {imbalance_ratio:.2f}:1. This can lead to low precision, recall, and F1-score for the minority class.")

                # Identify categorical features for encoding
                categorical_features = [col for col in X.columns if X[col].dtype == 'object']
                X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
                
                # Label encode specific categorical features if necessary
                if 'salary' in X_encoded.columns:
                    le = LabelEncoder()
                    X_encoded['salary'] = le.fit_transform(X_encoded['salary'])
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=0, stratify=y
                )
                
                # Model Hyperparameters (Simplified for a quick demo)
                xgb_params = {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100}
                ctb_params = {'iterations': 100, 'learning_rate': 0.3, 'depth': 6, 'l2_leaf_reg': 3}

                # Train XGBoost
                xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss', random_state=0)
                xgb_model.fit(X_train, y_train)

                # Train CatBoost
                ctb_model = CatBoostClassifier(**ctb_params, verbose=0, random_state=0)
                ctb_model.fit(X_train, y_train)

                # Evaluation
                def evaluate_model(model, X_test_data, y_test_data):
                    y_pred = model.predict(X_test_data)
                    y_proba = model.predict_proba(X_test_data)[:, 1]

                    metrics = {
                        'Accuracy': accuracy_score(y_test_data, y_pred),
                        'Precision': precision_score(y_test_data, y_pred, zero_division=0),
                        'Recall': recall_score(y_test_data, y_pred, zero_division=0),
                        'F1-Score': f1_score(y_test_data, y_pred, zero_division=0),
                        'ROC-AUC': roc_auc_score(y_test_data, y_proba)
                    }
                    return metrics
                
                xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
                ctb_metrics = evaluate_model(ctb_model, X_test, y_test)

                results_df = pd.DataFrame({
                    'XGBoost': xgb_metrics,
                    'CatBoost': ctb_metrics
                })
                
                return results_df, xgb_model, ctb_model, X_train

            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                return None, None, None, None

        results_df, xgb_model, ctb_model, X_train = train_and_evaluate_models(st.session_state['data'], feature_columns, target_column)
        
        if results_df is not None:
            st.session_state['results_df'] = results_df
            st.session_state['xgb_model'] = xgb_model
            st.session_state['ctb_model'] = ctb_model
            st.session_state['X_train_columns'] = X_train.columns
            st.success("Models trained successfully! You can now view results and make predictions.")


# --- Display Results Section (Conditional) ---
if 'results_df' in st.session_state:
    st.header("3. Model Performance Results")
    st.markdown("Here is a comparison of the trained models based on your selected features.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Metrics Table")
        st.dataframe(st.session_state['results_df'].style.highlight_max(axis=1, color='green'))

    with col2:
        st.subheader("Model Comparison Chart")
        st.bar_chart(st.session_state['results_df'].T)

    # --- Prediction Section (Conditional) ---
    st.header("4. Make a Prediction")
    st.markdown("Enter values for the features below to get a prediction from the trained models.")
    
    with st.form(key='prediction_form'):
        input_dict = {}
        for col in feature_columns:
            col_type = st.session_state['data'][col].dtype
            if col_type == 'object':
                unique_values = st.session_state['data'][col].unique().tolist()
                input_dict[col] = st.selectbox(f"Select a value for **{col}**:", unique_values)
            else:
                min_val = float(st.session_state['data'][col].min())
                max_val = float(st.session_state['data'][col].max())
                default_val = float(st.session_state['data'][col].median())
                input_dict[col] = st.number_input(f"Enter a value for **{col}**:", min_val, max_val, default_val)
        
        predict_button = st.form_submit_button("Predict Turnover", use_container_width=True)

    if predict_button:
        # Prepare the input data for prediction
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess input data to match training data format
        categorical_features = [col for col in input_df.columns if input_df[col].dtype == 'object']
        input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
        
        # Ensure columns match the training data
        missing_cols = set(st.session_state['X_train_columns']) - set(input_encoded.columns)
        for c in missing_cols:
            input_encoded[c] = 0
        input_encoded = input_encoded[st.session_state['X_train_columns']]
        
        # Make predictions
        xgb_pred = st.session_state['xgb_model'].predict(input_encoded)[0]
        ctb_pred = st.session_state['ctb_model'].predict(input_encoded)[0]

        # Map numerical predictions back to original labels
        prediction_map = {0: 'No', 1: 'Yes'}
        
        # Get the inverse mapping for better display
        inverse_map = {0: 'stay', 1: 'leave'}
        st.subheader("Prediction Results")
        st.success(f"**XGBoost Prediction:** The employee is likely to **{inverse_map[xgb_pred]}**.")
        st.success(f"**CatBoost Prediction:** The employee is likely to **{inverse_map[ctb_pred]}**.")

        st.markdown("---")
        st.markdown("**Note:** A 'Yes' prediction indicates the model believes there is a high likelihood of turnover.")
