import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Charity Donor Prediction",
    page_icon="",
    layout="centered"
)

st.title("Charity Donor Prediction")
st.write("Enter the details of the individual to predict if they are a potential donor or not.")

# ----------------------------------
# Load model and scaler
# ----------------------------------
@st.cache_resource
def load_artifacts():
    with open("New_RFmodel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("New_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# ----------------------------------
# Feature inputs
# ----------------------------------

mappings = {
    'workclass': {' Federal-gov': 0, ' Local-gov': 1, ' Private': 2, ' Self-emp-inc': 3, ' Self-emp-not-inc': 4, ' State-gov': 5, ' Without-pay': 6},
    'education_level': {' 10th': 0, ' 11th': 1, ' 12th': 2, ' 1st-4th': 3, ' 5th-6th': 4, ' 7th-8th': 5, ' 9th': 6, ' Assoc-acdm': 7, ' Assoc-voc': 8, ' Bachelors': 9, ' Doctorate': 10, ' HS-grad': 11, ' Masters': 12, ' Preschool': 13, ' Prof-school': 14, ' Some-college': 15},
    'marital-status': {' Divorced': 0, ' Married-AF-spouse': 1, ' Married-civ-spouse': 2, ' Married-spouse-absent': 3, ' Never-married': 4, ' Separated': 5, ' Widowed': 6},
    'occupation': {' Adm-clerical': 0, ' Armed-Forces': 1, ' Craft-repair': 2, ' Exec-managerial': 3, ' Farming-fishing': 4, ' Handlers-cleaners': 5, ' Machine-op-inspct': 6, ' Other-service': 7, ' Priv-house-serv': 8, ' Prof-specialty': 9, ' Protective-serv': 10, ' Sales': 11, ' Tech-support': 12, ' Transport-moving': 13},
    'relationship': {' Husband': 0, ' Not-in-family': 1, ' Other-relative': 2, ' Own-child': 3, ' Unmarried': 4, ' Wife': 5},
    'sex': {' Female': 0, ' Male': 1},
    'income': {'<=50K': 0, '>50K': 1}
}

feature_inputs = {
    'age': st.number_input("Age", min_value=0, max_value=100, value=30),
    'workclass': st.selectbox("Workclass", options=[ ' Federal-gov', ' Local-gov', ' Private', ' Self-emp-inc',' Self-emp-not-inc', ' State-gov', ' Without-pay']),
    'education_level': st.selectbox("Education Level", options=[' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th',' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate',' HS-grad', ' Masters', ' Preschool', ' Prof-school',' Some-college']),
    'marital-status': st.selectbox("Marital Status", options=[' Divorced', ' Married-AF-spouse', ' Married-civ-spouse',' Married-spouse-absent', ' Never-married', ' Separated',' Widowed']),
    'occupation': st.selectbox("Occupation", options=[' Adm-clerical', ' Armed-Forces', ' Craft-repair',' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners',' Machine-op-inspct', ' Other-service', ' Priv-house-serv',' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support',' Transport-moving']),
    'relationship': st.selectbox("Relationship", options=[' Husband', ' Not-in-family', ' Other-relative', ' Own-child',' Unmarried', ' Wife']),
    'sex': st.selectbox("Sex", options=[' Female', ' Male']),
    'capital-gain': st.number_input("Capital Gain", min_value=0, value=0),
    'capital-loss': st.number_input("Capital Loss", min_value=0, value=0),
    'hours-per-week': st.number_input("Hours per Week", min_value=0, max_value=168, value=40),
    'income': st.selectbox("Income", options=['<=50K', '>50K'])    
}

# Maintain correct feature order
feature_names = list(feature_inputs.keys())
input_values = [feature_inputs[f] for f in feature_names]

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Is this person a potential donor?"):
    processed_values = []
    for feature in feature_names:
        val = feature_inputs[feature]
        if feature in mappings:
            # Look up the integer value for the string selected by the user
            val = mappings[feature][val]
        processed_values.append(val)
    input_array = np.array(processed_values).reshape(1, -1)


    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = model.predict(scaled_input)
    donor="Yes" if prediction[0] == 1 else "No"

    st.success(f"**{donor}**")