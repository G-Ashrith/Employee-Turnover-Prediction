# 🚀 Employee Turnover Prediction Web App | A Customizable ML Framework

## 🌐 Overview
This project is an interactive, web-based tool for predicting employee turnover. It goes beyond a traditional Jupyter notebook by providing a flexible and user-friendly interface built with **Streamlit**. The application allows anyone to upload a custom HR dataset, select features, and train powerful machine learning models—**XGBoost** and **CatBoost**—on the fly to predict employee attrition.

The prediction of employee turnover is a critical task for organizations, as it directly impacts productivity and incurs significant costs related to recruitment and training. This app addresses this challenge by providing a framework that is both powerful and accessible.

---

## 🌟 Problem
The core challenge in building a generalized model is **adaptability**. Traditional machine learning scripts are often hardcoded to a specific dataset structure, making them brittle when new data is introduced.  
This project solves that problem by:

- **Dynamic UI Generation**: Automatically inferring data types and creating the appropriate input widgets for any given dataset.  
- **On-the-fly Training**: Allowing users to select their own features and re-train the models directly within the web app.  
- **Robustness**: Handling common issues like class imbalance and automatically converting non-numerical target variables.  

---

## 🚀 App Features
- **File Upload**: Upload your own employee dataset in CSV format.  
- **Dynamic Feature Selection**: Choose which columns to use as features and which to use as the target variable.  
- **Model Training**: Train XGBoost and CatBoost classifiers on your selected data with a single click.  
- **Performance Evaluation**: Instantly view and compare model performance using key classification metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
- **Live Prediction**: Use the trained models to get real-time predictions by entering custom values for each feature.  

---

## 📚 Repository Contents
- **app.py**: The main Streamlit web application that runs the entire project. This single file contains the user interface, data handling, and calls to the other modules.  
- **data_handler.py**: A module dedicated to loading data and handling the dynamic selection of features and target variables from the user interface.  
- **model_trainer.py**: A module that encapsulates the entire model training and evaluation pipeline, including data preprocessing and performance metric calculation.  
- **predictor.py**: A module containing the logic for making predictions on new data, ensuring the input is correctly preprocessed to match the trained model's format.  
- **HR.csv**: A sample dataset used for demonstration and testing purposes.  
- **requirements.txt**: A file listing all the necessary Python libraries for the project, ensuring a smooth deployment process.  

---

## 🚀 How to Run the App Locally

**1. Clone this Repository:**

git clone https://github.com/G-Ashrith/Employee-Turnover-Prediction.git

**2. Navigate to the Project Directory**
cd Employee-Turnover-Prediction

**3. Install Dependencies** 
pip install -r requirements.txt

**4. Run the App** 
streamlit run app.py


**🌐 Deployment**
   This application is deployed and publicly accessible via Streamlit Community Cloud.

   Live Demo: https://g-ashrith-employee-turnover-prediction-streamlitwebapp-kgqpdq.streamlit.app/
   Web App Screenshot: ![Web App Screenshot](Screenshots/webapp%20ss.png)
  

