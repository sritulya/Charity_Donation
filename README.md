# Charity_Donation

### 🎯 Finding Donors for CharityML
A Machine Learning project designed to help a fictitious non-profit, CharityML, identify individuals most likely to donate. This project uses supervised learning models to predict potential donors based on 1994 U.S. Census data.

### 📌 Project Overview
CharityML needs to reach out to potential donors more efficiently. Instead of sending mail to every person in a database, this model helps them target individuals who demonstrate a high likelihood of having a high income and a profile that aligns with charitable giving.

### 🛠️ Tech Stack
Language: Python 3.x

Web Framework: Streamlit (for the UI)

ML Libraries: Scikit-Learn, Pandas, NumPy

Visualization: Matplotlib, Seaborn

### 📂 Repository Contents
app.py: Streamlit application file providing the user interface for real-time predictions.

find_donors.ipynb: Jupyter Notebook documenting the data cleaning, exploration, and model training process.

census.csv: The dataset containing social and economic features.

New_RFmodel.pkl: The saved Random Forest Classifier model.

New_scaler.pkl: The saved MinMaxScaler used to normalize numerical features.

### ⚙️ How It Works
Data Preprocessing: Categorical variables (like workclass and occupation) are converted into numerical values using Label Encoding.

Feature Scaling: Numerical features (like capital-gain) are scaled to a range of 0 to 1 to ensure the model isn't biased by large numbers.

Prediction: The model analyzes features such as age, education level, and financial status to classify the individual as a Potential Donor or Not a Potential Donor.

### 🚀 Getting Started
### Prerequisites

### Ensure you have the following installed:
Python 3.8+
pip

### Installation:

Clone the repo:
git clone https://github.com/YOUR_USERNAME/charity-donor-prediction.git

Install the required libraries:
pip install streamlit pandas numpy scikit-learn

Run the Streamlit app:
streamlit run app.py

### 📈 Model Performance
The project evaluates several models (Logistic Regression, Decision Trees, Random Forest) before selecting the best performer based on the F-beta score, which balances precision and recall to minimize "false positives"—ensuring the charity doesn't waste resources on unlikely donors.
