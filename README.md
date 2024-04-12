# Kijiji_Housing

Overview:

This project aims to predict the rental housing prices in Ontario using machine learning techniques. The prediction model is based on several key features extracted from rental listings on the Kijiji website, including the number of bedrooms, bathrooms, size (in square feet), type of property, and the name of the Census Subdivision (CSDNAME).

Dataset:

The dataset used for training and testing the machine learning model is sourced from a CSV file containing information gathered from rental listings on Kijiji. The dataset includes various attributes of rental properties, such as the number of bedrooms, bathrooms, size, type, and CSDNAME.

Machine Learning Model:

We employ a supervised learning approach to train the machine learning model. Specifically, we use a linear regression algorithm to predict rental housing prices based on the selected features. The model is trained on a portion of the dataset and evaluated for its performance on unseen data to ensure its effectiveness in making accurate predictions.

How to Use: To use the prediction model, follow these steps:

Clone this repository to your local machine. Install the necessary dependencies specified in the requirements.txt file. Run the Streamlit app by executing the app.py script. In the Streamlit app, select the desired features, such as the number of bedrooms, bathrooms, size, type, and CSDNAME. The app will display the predicted rental housing price based on the selected features.

Dependencies: The project requires the following dependencies:

Streamlit Pandas Scikit-learn
