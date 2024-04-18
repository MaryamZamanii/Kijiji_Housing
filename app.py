import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import folium


# Load the dataset with a specified encoding
data = pd.read_csv('ontario_EDA.csv', encoding='latin1')

#page1
def dashboard():
    st.image('Logo2.png', width=350)

    # Abstract section
    st.subheader("💡 Abstract:")
    inspiration = '''
    The Community Data Project on rental housing from the Kijiji website provided a rich learning experience, combining data cleaning, data analysis, machine learning, and collaboration. Through this project, we gained insights into rental housing trends and applied data science techniques to extract valuable information for stakeholders.
    '''
    st.markdown(f'<div style="background-color:#c6ffb3; padding: 15px; border-radius: 10px;"><p style="color:#000000 ;font-size:16px;">{inspiration}</p></div>', unsafe_allow_html=True)

    # Project overview section
    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
    Our project analyzes rental housing data from the Kijiji website, uncovering insights and trends to support decision-making in the community.
    '''

    st.markdown(f'<div style="background-color:#c6ffb3; padding: 15px; border-radius: 10px;"><p style="color:#000000;font-size:16px;">{what_it_does}</p></div>', unsafe_allow_html=True)


# Page 2: Exploratory Data Analysis (EDA)

def exploratory_data_analysis():
  data = pd.read_csv("All_facilities_merged1.csv")

# Convert 'Price' column to numeric format
  data['Price'] = data['Price'].str.replace('$', '').str.replace(',', '').astype(float)

# Filter the dataset for small communities with population < 10,000
  small_communities = data[data['Population'] < 10000]

# Group by healthcare_count and calculate the average price
  avg_price_by_healthcare = small_communities.groupby('healthcare_count')['Price'].mean().reset_index()

# Streamlit app
  st.subheader('Average Rental Price based on Healthcare Facilities in Small Communities (<10,000 population)')

# Create a bar plot using Plotly Express
  fig = px.bar(avg_price_by_healthcare,
             x='healthcare_count',
             y='Price',
             labels={'Price': 'Average Price ($)', 'healthcare_count': 'Number of Healthcare Facilities'},
             color_discrete_sequence=['lightgreen']
            )

# Update layout
  fig.update_layout(xaxis_tickangle=0,
                  xaxis=dict(title='Number of Healthcare Facilities'),
                  yaxis=dict(title='Average Price ($)'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                 )

# Show the plot using Streamlit
  st.plotly_chart(fig)


  st.subheader('Relationship Between Healthcare Facilities and Rental Prices in Small Community')

# Create a scatter plot using Plotly Express
  fig = px.scatter(small_communities,
                 x='healthcare_count',
                 y='Price',
                 trendline='ols',  # Add linear regression trendline
                 labels={'Price': 'Rental Price ($)', 'healthcare_count': 'Number of Healthcare Facilities'},
                 color_discrete_sequence=['salmon']
                )

# Update layout
  fig.update_layout(xaxis=dict(title='Number of Healthcare Facilities'),
                  yaxis=dict(title='Rental Price ($)'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                 )

# Show the plot using Streamlit
  st.plotly_chart(fig)

  # Streamlit app
  st.subheader('Distribution of Rental Prices Across Different Types of Educational Facilities for small community')

# Group by education_facility_type and calculate the average price
  avg_price_by_education = small_communities.groupby('education_facility_type')['Price'].mean().reset_index()

# Create a bar chart using Plotly Express
  fig = px.bar(avg_price_by_education,
             x='education_facility_type',
             y='Price',
             labels={'Price': 'Average Rental Price ($)', 'education_facility_type': 'Type of Educational Facility'},
             color_discrete_sequence=['orange']
            )

# Update layout
  fig.update_layout(xaxis_tickangle=45,
                  xaxis=dict(title='Type of Educational Facility'),
                  yaxis=dict(title='Average Rental Price ($)'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                 )

# Show the plot using Streamlit
  st.plotly_chart(fig)

# Streamlit app

  recreational_facilities_to_include = [ "['arena', 'trail']",
                                      "['pool', 'trail']",
                                      "['sports field']",
                                      "['park']",
                                      "['community centre']"]

# Filter the data for the specified recreational facilities
  data_filtered = data[data['recreational_type'].isin(recreational_facilities_to_include)]

# Streamlit app
  st.subheader('Distribution of Rental Prices based on most popular Recreational Facilities for small community')

# Create a box plot using Plotly Express
  fig = px.box(data_filtered,
             x='recreational_type',
             y='Price',
             labels={'Price': 'Rental Price ($)', 'recreational_type': 'Recreational Facility'},
             width=1200,  # Adjust the width of the plot
             height=900   # Adjust the height of the plot
            )

# Update layout
  fig.update_layout(xaxis=dict(title='Recreational Facility'),
                  yaxis=dict(title='Rental Price ($)'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                 )

# Show the plot using Streamlit
  st.plotly_chart(fig)
  recreational_facilities_to_include = ["['arena', 'trail']", "['pool', 'trail']","['sports field']","['park']","['community centre']"]

# Filter the data for the specified recreational facilities
  data_filtered = data[data['recreational_type'].isin(recreational_facilities_to_include)]

# Calculate the average rental price for each recreational facility
  avg_price_by_facility = data_filtered.groupby('recreational_type')['Price'].mean().reset_index()

# Streamlit app
  st.subheader('Average Rental Prices for most popular Recreational Facilities for small community')

# Create a grouped bar chart using Plotly Express
  fig = px.bar(avg_price_by_facility,
             x='recreational_type',
             y='Price',
             labels={'Price': 'Average Rental Price ($)', 'recreational_type': 'Recreational Facility'},
             color='recreational_type',
             color_discrete_map={'park': 'blue', 'sports field': 'green', 'gym': 'red', 'playground': 'orange', 'community center': 'purple'}
            )

# Update layout
  fig.update_layout(xaxis=dict(title='Recreational Facility'),
                  yaxis=dict(title='Average Rental Price ($)'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                  legend_title='Recreational Facility'
                 )

# Show the plot using Streamlit
  st.plotly_chart(fig)

  # Filter the dataset for culture facilities
  culture_data = data[data['cultural_facilityname'].notnull()]

# Define the cultural facilities to include in the pie chart
  cultural_facilities_to_include = ["['museum']","['art or cultural centre', 'museum', 'heritage or historic site']",
                                  "['library or archives']",
                                  "['miscellaneous']",
                                  "['museum', 'heritage or historic site', 'gallery']",
                                  "['theatre/performance and concert hall']"]

# Filter the data for the specified cultural facilities
  filtered_data = culture_data[culture_data['cultural_facilityname'].apply(lambda x: x in cultural_facilities_to_include)]

# Count the occurrences of each cultural facility
  culture_count = filtered_data['cultural_facilityname'].value_counts().reset_index()
  culture_count.columns = ['Cultural Facility', 'Count']

# Streamlit app
  st.subheader('Pie Chart of most popular Cultural Facilities for small community')

# Create a pie chart using Plotly Express
  fig = px.pie(culture_count,
             values='Count',
             names='Cultural Facility',
             title='Distribution of Selected Cultural Facilities',
             hole=0.4,  # Size of the center hole
             color_discrete_sequence=px.colors.qualitative.Pastel  # Color palette
            )

# Show the plot using Streamlit
  st.plotly_chart(fig)


# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Kijiji Rental Price Prediction")
    st.write("Enter the details of the property to predict its rental price:")

    # Input fields for user to enter data
    st.subheader("Property Details:")
    unique_locations = data['CSDNAME'].unique()
    location = st.selectbox("Location", unique_locations)
    size = st.slider("Size (sqft)", 300, 5000, 1000)
    bedrooms = st.selectbox('Number of Bedrooms', options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], format_func=lambda x: f"{x:.1f}", index=1)
    bathrooms = st.selectbox('Number of Bathrooms', options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], format_func=lambda x: f"{x:.1f}", index=1)
    property_type = st.radio('Type', data['Type'].unique())

    if st.button("Predict", key='predict_button'):
        # Load the trained model including preprocessing
        model = joblib.load('gradient_boost_regressor_model.pkl')

        # Prepare input data as a DataFrame to match the training data structure
        input_df = pd.DataFrame({
            'Type': [property_type],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Size': [size],
            'CSDNAME': [location]
        })

        # Make prediction
        prediction = model.predict(input_df)

        # Display the prediction
        st.write(f"""
        <div style="background-color:#E5F4E3; padding: 8px; border-radius: 8px;">
            <h3 style="color:#009900;">Predicted Rental Price: ${prediction[0]:,.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Chat Box
    st.subheader("Have questions? Ask our Assistant!")
    chatbot_url = "https://hf.co/chat/assistant/6618ba66044cc6a08eefa689"
    st.markdown(f'<iframe src="{chatbot_url}" width="500" height="500"></iframe>', unsafe_allow_html=True)

# Page 4: Community Mapping
def community_mapping():
    st.title("Communities Map")
    st.write("<span style='font-weight:bold; color:black;'>Each point represents a rental property listing on the Kijiji website</span>", unsafe_allow_html=True)
    st.write("<span style='font-weight:bold; color:green;'>Color Of Each Point ➔ Population</span>", unsafe_allow_html=True)
    st.write("<span style='font-weight:bold; color:green;'>Size of Of Each Point ➔ Price</span>", unsafe_allow_html=True)
    geodata = pd.read_csv("ontario_EDA.csv")

    # Set your Mapbox token
    px.set_mapbox_access_token('pk.eyJ1IjoibXlhc2thbWFzIiwiYSI6ImNrbHV5NWh4NzBlMm4ydnBnYm5nNTR1OWUifQ.BmR8YJHx7QolurVQoK-WCg')

    # Create the map using Plotly Express
    fig = px.scatter_mapbox(geodata,
                            lat='Latitude',
                            lon='Longitude',
                            color='Population',  # Color points by population, or choose another column
                            size='Price',  # Size points by price, or choose another column
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10,
                            hover_name='Type',  # Display property type when hovering over points
                            hover_data={'Price': True, 'Population': True, 'Latitude': False, 'Longitude': False, 'CSDNAME': True, 'Bedrooms': True})

    fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style
    st.plotly_chart(fig)



    # Page 5: small Communities Mapping
def small_community_mapping():
    st.subheader("Small Community Map: Population <10000")
    st.write("<span style='font-weight:bold; color:black;'>'Each point represents a rental property listing on the Kijiji website for small community'</span>", unsafe_allow_html=True)
    st.write("<span style='font-weight:bold; color:green;'>'Color Of Each Point ➔ Population'</span>", unsafe_allow_html=True)
    st.write("<span style='font-weight:bold; color:green;'>'Size Of Each Point ➔ Price'</span>", unsafe_allow_html=True)
    geodata = pd.read_csv("ontario_EDA.csv")

    # Filter communities with population less than 10,000
    geodata_filtered = geodata[geodata['Population'] < 10000]

    # Create the map using Plotly Express
    fig = px.scatter_mapbox(geodata_filtered,
                            lat='Latitude',
                            lon='Longitude',
                            color='Population',  # Color points by population, or choose another column
                            size='Price',  # Size points by price, or choose another column
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            size_max=15,
                            zoom=10,
                            hover_name='Type',  # Display property type when hovering over points
                            hover_data={'Price': True, 'Population': True, 'Latitude': False, 'Longitude': False,'CSDNAME': True, 'Bedrooms':True})

    fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style
    st.plotly_chart(fig)

def Lookerstudio():

    # Embedding Google Map using HTML iframe
    st.markdown("""
<iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/93a94dce-e9e0-4cd1-bee7-d358dfd6424c/page/L9WpD" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
""", unsafe_allow_html=True)

# Main App Logic
def main():
    st.sidebar.title("Kijiji Community App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Community Mapping", "Small Community Mapping","Lookerstudio"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Community Mapping":
        community_mapping()
    elif app_page == "Small Community Mapping":
        small_community_mapping()
    elif app_page == "Lookerstudio":
        Lookerstudio()

if __name__ == "__main__":
    main()
