import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df_ML = pd.read_csv("ontario_EDA.csv")

# Set page title and favicon
st.set_page_config(page_title="Ontario Kijiji Housing", page_icon="🏠")

# Set up the Streamlit app with a custom background color and text colors
st.markdown(
    """
    <style>
    body {
        background-color: #d4f4dd; /* Light green */
        color: #333333; /* Dark blue */
    }
    .feature-box {
        background-color: #c7ecee; /* Light blue */
        padding: 10px ;
        border-radius: 10px ;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0.1,0.9);
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        width: 150px; /* Adjust the width of the feature boxes */
        text-align: center; /* Center align text */
    }
    .feature-box:nth-child(even) {
        background-color: #a9cce3; /* Light blue for alternate boxes */
    }
    .title-text {
        font-size: 18px;
        color: red; /* Red title text */
        font-weight: bold; /* Make title text bold */
    }
    .header-text {
        font-size: 24px;
        font-weight: bold;
        color: red; /* Red header text */
        margin-bottom: 20px;
    }
    .subheader-text {
        font-size: 16px;
        color: red !important; /* Red subheader text */
    }
    .selected-feature {
        font-weight: bold; /* Make selected feature text bold */
    }
    .predicted-quantity {
        font-weight: bold; /* Make predicted quantity text bold */
        color: #000000; /* Black text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to preprocess the data and make predictions
def make_predictions(model, features):
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

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    prediction = model.predict(
        ct.transform([[features['CSDNAME'], features['Bedrooms'], features['Bathrooms'], features['Size'], features['Type']]]))
    return prediction

def main():
    st.title('Ontario Kijiji Housing')

    # Sidebar
    st.sidebar.subheader('Select Features')
    Size = st.sidebar.slider('Size (sq ft)', 300, 9000, value=1000)
    Bedrooms = st.sidebar.selectbox('Bedrooms', options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], format_func=lambda x: f"{x:.1f}", index=1)
    Bathrooms = st.sidebar.selectbox('Bathrooms', options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], format_func=lambda x: f"{x:.1f}", index=1)
    CSDNAME = st.sidebar.selectbox('CSDNAME', df_ML['CSDNAME'].unique())
    Type = st.sidebar.radio('Type', df_ML['Type'].unique())

    # User Input Parameters
    st.subheader('Selected Features:')
    selected_features = {
        'Size (sq ft)': Size,
        'Bedrooms': f"{Bedrooms:.1f}",
        'Bathrooms': f"{Bathrooms:.1f}",
        'CSDNAME': CSDNAME,
        'Type': Type
    }
    selected_features_df = pd.DataFrame.from_dict(selected_features, orient='index', columns=['Value']).T
    st.write(selected_features_df.style.set_properties(**{'font-weight': 'bold'}))

    # Train a Linear Regression model
    linear_model = LinearRegression()

    # Make predictions
    prediction = make_predictions(linear_model, {'CSDNAME': CSDNAME, 'Bedrooms': Bedrooms, 'Bathrooms': Bathrooms, 'Size': Size, 'Type': Type})

    # Display Prediction
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader('Predicted Price:')
    st.write(f"<div class='feature-box'><b>${prediction[0]:,.2f}</b></div>", unsafe_allow_html=True)

    # Embedding Chat Box using HTML iframe
    st.write('Welcome to our Kijiji Rental Housing Assistant!')
    chatbot_url = "https://hf.co/chat/assistant/6618ba66044cc6a08eefa689"
    st.markdown(f'<iframe src="{chatbot_url}" width="700" height="500"></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
