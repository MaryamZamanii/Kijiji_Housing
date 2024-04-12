import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df_ML = pd.read_csv("ontario_EDA.csv")

# Set page title and favicon
st.set_page_config(page_title="Ontario Kijiji Housing", page_icon="🏠")

# Set up the Streamlit app with a custom background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("Ontario Kijiji Housing")
st.write("Predict the price of a house based on selected features.")

# Sidebar
st.sidebar.header('User Input')
st.sidebar.subheader('Select Features')
Size = st.sidebar.slider('Size (sq ft)', 300, 2000, 4200)
Bedrooms = st.sidebar.selectbox('Bedrooms', options=[0.5, 1.0,1.5, 2.0,2.5,3.0,3.5,4.0,4.5, 5.0])
Bathrooms = st.sidebar.selectbox('Bathrooms', options=[0.5, 1.0,1.5, 2.0,2.5,3.0,3.5,4.0])
CSDNAME = st.sidebar.selectbox('CSDNAME', df_ML['CSDNAME'].unique())
Type = st.sidebar.selectbox('Type', df_ML['Type'].unique())

# User Input Parameters
st.subheader('Selected Features')
st.write("**Selected Features:**", end=' ')
st.write(
    f"<div class='feature-box'>Size (sq ft): {Size} sq ft</div>"
    f"<div class='feature-box'>Bedrooms: {Bedrooms}</div>"
    f"<div class='feature-box'>Bathrooms: {Bathrooms}</div>"
    f"<div class='feature-box'>CSDNAME: {CSDNAME}</div>"
    f"<div class='feature-box'>Type: {Type}</div>",
    unsafe_allow_html=True
)

# Preprocess the data
X = df_ML[['CSDNAME', 'Bedrooms', 'Bathrooms', 'Size', 'Type']].values
y = df_ML['Price'].values

# One-hot encode CSDNAME and Type
ct = ColumnTransformer(
    [('onehot', OneHotEncoder(), [0, 4])], remainder='passthrough')
X = ct.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
prediction = linear_model.predict(
    ct.transform([[CSDNAME, Bedrooms, Bathrooms, Size, Type]]))

# Display Prediction
st.markdown("<hr>", unsafe_allow_html=True)
st.write("**Predicted Price:**")
st.write(f"<div class='feature-box'><b>${prediction[0]:,.2f}</b></div>", unsafe_allow_html=True)
