import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from io import StringIO
from zipfile import BadZipFile

# Internal CSS for background color and custom styles
st.markdown("""
    <style>
        /* Background color for the main app */
        body {
            background-color: #f0f2f6;
            color: #333333;
        }

        /* Sidebar background color and text styling */
        section[data-testid="stSidebar"] {
            background-color: #2e4057;
            color: white;
        }

        /* Sidebar text color */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: white;
        }

        /* Style for main app titles and headers */
        h1, h2, h3 {
            color: #2e4057;
        }

        /* Style for buttons */
        button[kind="primary"] {
            background-color: #2e4057;
            color: white;
        }

        button[kind="primary"]:hover {
            background-color: #3a506b;
            color: white;
        }

        /* Style for dataframe headers and table text */
        .stDataFrame th {
            background-color: #2e4057;
            color: white;
        }

        .stDataFrame td {
            color: #333333;
        }    
    </style>
    """, unsafe_allow_html=True)

# Load and display the team logo
st.sidebar.image("teamlogo.png", use_column_width=True)  # Add the logo at the top of the sidebar

# Title for the web app
st.title("FINANCIAL TRANSACTION FRAUD DETECTION APP")

# Sidebar for user to upload data or input data as text
st.sidebar.header("Upload or Input Data")
data_option = st.sidebar.selectbox("Choose how you want to input data:", ["Upload CSV", "Upload Excel", "Input Data Manually"])

# Initialize an empty dataframe
df = None

# Option 1: Upload CSV File
if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("## Dataset Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")

# Option 2: Upload Excel File
elif data_option == "Upload Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xls", "xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')  # Specify the engine
            st.write("## Dataset Preview")
            st.dataframe(df.head())
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except BadZipFile:
            st.error("The uploaded file is not a valid Excel file. Please upload a .xls or .xlsx file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while reading the file: {e}")

# Option 3: Manually Input Data as Text
elif data_option == "Input Data Manually":
    st.write("## Enter Data as Text")
    
    input_text = st.text_area("Paste your data here (CSV format)", value="", height=200)
    
    if input_text:
        input_data = StringIO(input_text)
        df = pd.read_csv(input_data)
        
        st.write("## Dataset Preview")
        st.dataframe(df.head())

# Proceed if a dataset is provided (either uploaded or manually entered)
if df is not None:
    # Display basic dataset info
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Checking for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Handling missing values
    if st.button("Impute Missing Values"):
        impute = KNNImputer()
        for i in df.select_dtypes(include="number").columns:
            df[i] = impute.fit_transform(df[[i]])
        st.write("Missing values filled using KNN Imputation")
        st.write(df.isnull().sum())

    # Encode categorical variables
    st.write("### Categorical Encoding")
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    st.write("### Data after Label Encoding")
    st.dataframe(df.head())

    # Split data into features and target
    Y = df['is_fraud']
    X = df.drop(columns=["is_fraud"])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Feature selection using Variance Threshold
    var_thres = VarianceThreshold(threshold=2.0)
    var_thres.fit(X_train)
    constant_columns = [column for column in X_train.columns if column not in X_train.columns[var_thres.get_support()]]
    X_train.drop(constant_columns, axis=1, inplace=True)
    X_test.drop(constant_columns, axis=1, inplace=True)

    # Streamlit model section
    st.subheader("Model Training and Evaluation")

    # Select the noise level for the target variable
    noise_level = st.slider("Select Noise Level (%)", 0.0, 100.0, 20.0) / 100.0  # Convert to fraction

    # Introduce noise to the target variable (Y)
    noisy_Y = Y.copy()  # Create a copy of the original target variable
    for i in range(len(noisy_Y)):
        if np.random.random() < noise_level:
            noisy_Y.iloc[i] = 1 - noisy_Y.iloc[i]  # Assuming binary classification with labels 0 and 1

    # Split the data into training and testing sets using the noisy target variable
    X_train, X_test, y_train, y_test = train_test_split(X, noisy_Y, test_size=0.25, random_state=0)

    # Naive Bayes model
    nb_model = GaussianNB()

    # Handle NaN values in the feature sets
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # Convert y_train to discrete values
    y_train = y_train.astype(int)

    # Train the model
    nb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nb_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    st.write(f"Naive Bayes Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.write("Please upload a CSV or Excel file or input data to proceed.")import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# Internal CSS for background color and custom styles
st.markdown("""
    <style>
        /* Background color for the main app */
        body {
            background-color: #f0f2f6;
            color: #333333;
        }

        /* Sidebar background color and text styling */
        section[data-testid="stSidebar"] {
            background-color: #2e4057;
            color: white;
        }

        /* Sidebar text color */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: white;
        }

        /* Style for main app titles and headers */
        h1, h2, h3 {
            color: #2e4057;
        }

        /* Style for buttons */
        button[kind="primary"] {
            background-color: #2e4057;
            color: white;
        }

        button[kind="primary"]:hover {
            background-color: #3a506b;
            color: white;
        }

        /* Style for dataframe headers and table text */
        .stDataFrame th {
            background-color: #2e4057;
            color: white;
        }

        .stDataFrame td {
            color: #333333;
        }
            
    </style>
    """, unsafe_allow_html=True)

# Load and display the team logo
st.sidebar.image("teamlogo.png", use_column_width=True)  # Add the logo at the top of the sidebar

# Title for the web app
st.title("FINANCIAL TRANSACTION FRAUD DETECTION APP")

# Sidebar for user to upload data or input data as text
st.sidebar.header("Upload or Input Data")
data_option = st.sidebar.selectbox("Choose how you want to input data:", ["Upload CSV", "Input Data Manually"])

# Initialize an empty dataframe
df = None

# Option 1: Upload CSV File
if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## Dataset Preview")
        st.dataframe(df.head())

# Option 2: Manually Input Data as Text
elif data_option == "Input Data Manually":
    st.write("## Enter Data as Text")
    
    # Sample input
    #st.write("**Example Input (Comma-separated, including header):**")
    #st.write("cc_num,amt,zip,lat,long,city_pop,unix_time,merch_lat,merch_long,is_fraud")
    #st.write("1234567890123456,200.0,12345,45.12,-93.5,1000,1638480000,45.50,-93.40,0")
    #st.write("9876543210987654,500.5,54321,40.71,-74.01,500000,1638480000,40.70,-74.00,1")
    
    # Text area to input data manually
    input_text = st.text_area("Paste your data here (CSV format)", value="", height=200)
    
    if input_text:
        # Convert the input text to a pandas dataframe
        from io import StringIO
        input_data = StringIO(input_text)
        df = pd.read_csv(input_data)
        
        st.write("## Dataset Preview")
        st.dataframe(df.head())

# Proceed if a dataset is provided (either uploaded or manually entered)
if df is not None:
    # Display basic dataset info
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Checking for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Handling missing values
    if st.button("Impute Missing Values"):
        impute = KNNImputer()
        for i in df.select_dtypes(include="number").columns:
            df[i] = impute.fit_transform(df[[i]])
        st.write("Missing values filled using KNN Imputation")
        st.write(df.isnull().sum())

    # Encode categorical variables
    st.write("### Categorical Encoding")
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    st.write("### Data after Label Encoding")
    st.dataframe(df.head())

    # Splitting dataset into training and testing sets
    if 'is_fraud' in df.columns:
        Y = df['is_fraud']
        X = df.drop(columns=["is_fraud"])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Train Naive Bayes model
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)

        # Predict on test set
        y_pred = nb_model.predict(X_test)

        # Display classification results
        #st.write("### Model Evaluation")
        #st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        #st.write("Classification Report:")
        #st.text(classification_report(y_test, y_pred))

        # Section for manual input to predict a single transaction
        st.write("## Predict Fraud for a Single Transaction")
        st.write("Enter the following transaction details:")

        # Create input fields dynamically based on the features in the dataset
        input_data = {}
        for col in X.columns:
            if df[col].dtype == 'object':
                input_data[col] = st.text_input(f"{col}")
            elif df[col].dtype == 'int64':
                input_data[col] = st.number_input(f"{col}", value=0)
            elif df[col].dtype == 'float64':
                input_data[col] = st.number_input(f"{col}", value=0.0)

        # Predict fraud for the manually entered transaction
        if st.button("Predict Fraud"):
            input_df = pd.DataFrame([input_data])

            # Encode categorical values of input data
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = label_encoder.transform(input_df[col])

            # Perform prediction
            prediction = nb_model.predict(input_df)[0]

            if prediction == 1:
                st.write("The transaction is **fraudulent**.")
            else:
                st.write("The transaction is **not fraudulent**.")

else:
    st.write("Please upload a CSV file or input data to proceed.")
