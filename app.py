



# Import necessary libraries
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.set_page_config(page_title="Medical Insurance Claim Prediction")

# Load the dataset
data = pd.read_csv("Medical-Insurance-Claim-Prediction-main\I_C_DATA.csv")

# Define the features and target variable
features = ['age', 'sex', 'bmi', 'steps', 'children', 'smoker', 'region', 'charges']
target = 'insuranceclaim'

# Create the sidebar for user input
st.header('Medical Insurance Claim Prediction')


age = st.slider('Age', 18, 100, 25)
sex = st.radio('Sex', ('Male', 'Female'))
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
steps = st.slider('Steps', 0, 20000, 10000)
children = st.slider('Children', 0, 5, 1)
smoker = st.radio('Smoker', ('Yes', 'No'))
region = st.radio('Region', ('Northeast', 'Northwest', 'Southeast', 'Southwest'))
charges = st.slider('Charges', 1000, 60000, 10000)

# Transform user input data
sex_map = {'Male': 0, 'Female': 1}
smoker_map = {'Yes': 1, 'No': 0}
region_map = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex_map[sex]],
    'bmi': [bmi],
    'steps': [steps],
    'children': [children],
    'smoker': [smoker_map[smoker]],
    'region': [region_map[region]],
    'charges': [charges]
})

# Train a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Create a predict button
if st.button('Predict'):
    # Make prediction
    prediction = clf.predict(user_input)

# Display the prediction
    if prediction[0] == 1:
        st.success('He Can Claim Medical Insurance')
    else:
        st.warning('He Cannot Claim Medical Insurance')

















