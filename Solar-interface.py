import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from PIL import Image
import streamlit.components.v1 as stc
import seaborn as sns 
import matplotlib.pyplot as plt
import plost


#Setting Page icon & configuration

img = Image.open("icon.jpg")
PAGE_CONFIG = {"page_title":"Solar Power App","page_icon":img,"layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


@st.cache_data
def caching():
    st.header("Enhancing Performance!")

st.cache_data.clear()    
    
    
# Creating menu

menu = ['🏡 Home', '📈 Data Visualization', '🎯 Generate Prediction','🖊 Comment Section']
choice = st.sidebar.radio('Menu',menu)


if choice == '🏡 Home':
    #st.title("SOLAR POWER GENERATION FORECASTING")
    html_temp = """
    <div style="background-color:#fb6c38;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">SOLAR POWER GENERATION FORECASTING </h1>
    </div>"""
    stc.html(html_temp)
    img = st.image("solimg.jpg")
    
    st.markdown("The objective of this project is to leverage machine learning techniques, such as **Linear Regression, k-nearest neighbor (KNN), Decision Tree and Random Forest Regressor**, compare the evaluation metrics of the models and chose the best one to predict solar power generation based on the dataset. *By achieving accurate predictions, this project aims to assist in efficient energy management and facilitate the integration of solar power into the existing power grid.*")
    
    st.header("Dataset Description")
    #st.subheader("")
    plant = st.selectbox("Select Plant to show dataset", options = ['Plant1', 'Plant2'])
    if plant == 'Plant1':
        gen1 = pd.read_csv('Plant_1_Generation_Data.csv')
        wet1 = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
        st.table(gen1.head())
        st.table(wet1.head())
    elif plant == 'Plant2':
        gen2= pd.read_csv('Plant_2_Generation_Data.csv')
        wet2= pd.read_csv('Plant_2_Weather_Sensor_Data.csv')
        st.table(gen2.head())
        st.table(wet2.head())
        
    st.write("")
    st.write("")
    
    st.image("Ideal_Graph.jpg", caption= 'Ideal Graph of Solar Power Generation')
    
    st.write("")
    st.write("")
    
    st.markdown("# *Advantages:*")
    st.write("")
    st.markdown("*Some of the factors for choosing the solar power generation are listed below.*")
    st.markdown(" ➼ Solar energy is available freely and conveniently in nature and it needs no mains supply.")
    st.markdown(" ➼ Solar generation plant can be installed in a few months while the conventional power plants take several years to build an electricity generation plant.")
    st.markdown(" ➼ Solar power is clean energy as it produces no air or water pollution. Also, there are no moving parts to create noise pollution. Unlike fossil fuels, no toxic emissions are released into the atmosphere during solar energy power generation.")
    st.markdown(" ➼ Solar power has less running cost that means once the capital investment is made, there is no need for continues purchase of fossil fuels as the solar energy is effectively free in nature.")
    
    
        
elif choice == '📈 Data Visualization':
    
    #data = pd.merge(gen1.drop(columns = ['PLANT_ID']), wet1.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
    data = pd.read_csv('data.csv')
    
    st.markdown("## *Welcome to Data Visualization:*")
    
    st.write("")
    st.write("")
    
    chart = st.selectbox('Choose a Visualization Type: ', ['Line Chart','Bar Chart', 'Histogram', 'Scatter Plot', 'Area Chart'])
    
    st.write("")
    st.write("")
    
    
    if chart == 'Line Chart':
        @st.cache_resource
        def line():
            #st.header("Faster loading!...")
            plost.line_chart(data, x= 'AMBIENT_TEMPERATURE', y= 'AC_POWER', 
            width=500, color = "#17d196", 
            title='AC power based on Ambient Temperature', legend='bottom')
        line()
        
        st.write("")
        st.write("")
        st.write("")
    
    elif chart == 'Bar Chart':
        @st.cache_resource
        def bar():
            plost.bar_chart(
            data,
            bar='IRRADIATION',
            value=['DC_POWER', 'AC_POWER'],
            group='value',
            stack=True,
            color='IRRADIATION',
            legend="top",)
        bar()
    
    
        st.write("")
        st.write("")
        st.write("")    
    
    elif chart == 'Histogram':
        @st.cache_resource
        def hist():
            plost.hist(
            data,
            x='IRRADIATION',
            y= 'DAILY_YIELD',
            aggregate='median',
            title='Irradiation vs Daily Yield', legend='bottom')
        hist()
    
    
        st.write("")
        st.write("")
        st.write("")
    
    elif chart == 'Scatter Plot':
        @st.cache_resource
        def sct():
            plost.scatter_chart(
            data,
            x='AMBIENT_TEMPERATURE',
            y=['DC_POWER', 'AC_POWER'],
            size='IRRADIATION',
            height=500)
        sct()
        
        st.write("")
        st.write("")
        st.write("")
    
    
    elif chart == 'Area Chart':
        @st.cache_resource
        def area():
            st.markdown("**The Ideal Graph of Solar Power Generation:**")
            st.write("")
            plost.area_chart(
            data,
            x=dict(field='DATE_TIME', timeUnit='hours'),
            y=dict(field='AC_POWER', aggregate='mean'),
            color='IRRADIATION',)
        area()
        
    st.cache_resource.clear()
        

elif choice == '🎯 Generate Prediction':
    # Feature vector 
    #X = data[['DAILY_YIELD', 'TOTAL_YIELD', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]
    #y = data['AC_POWER']
    
    st.subheader('Kindly fill in the information to make Predictions:')
    
    #taking user inputs for features
    name = st.text_input("Your Good Name?")
    if name:
        st.markdown(f"Greetings! **{name}**, Welcome to our App.")
        
        
    #algo = st.selectbox('Choose an Algorithm: ', ['Linear Regression','KNN', 'Decision Tree', 'Random Forest'])
    
    col1, col2 = st.columns(2)
    
    
    DAILY_YIELD = col1.number_input("Input Daily Yield: ")
    DAILY_YIELD = col2.number_input("Input Total Yield: ")
    AMBIENT_TEMPERATURE = col1.number_input("Input Ambient Temperature(°C): ")
    MODULE_TEMPERATURE = col2.number_input("Input Module Temperature(°C): ")
    IRRADIATION = st.slider("Input Irradiation(kW/m²): ",0.00,1.50)
    
    test_data = {'DAILY_YIELD': [DAILY_YIELD], 'TOTAL_YIELD': [DAILY_YIELD], 'AMBIENT_TEMPERATURE': [AMBIENT_TEMPERATURE],'IRRADIATION': [IRRADIATION]}
    
    #st.button('Predict')
    
    df_pred = pd.DataFrame(test_data)
    
    #loading the models
    #if algo == 'Linear Regression':
    #    model = joblib.load('LR.pkl')
    #    prediction = model.predict(df_pred)
        
        
    #elif algo == 'KNN':
    #    model = joblib.load('KNN.pkl')
    #    prediction = model.predict(df_pred)
        
        
    #elif algo == 'Decision Tree':
    model = joblib.load('Decision_Tree.pkl')
    prediction = model.predict(df_pred)
        
        
    #elif algo == 'Random Forest':
    #    model = joblib.load('Random_forest.pkl')
    #    prediction = model.predict(df_pred)
    
    
    #Making predictions
    if st.button('Predict'):
        
        with st.spinner('Model is working!...'):
            time.sleep(1.0)
            st.success("The AC Power Predicted is : {} MW".format(prediction))
        #st.snow()
        #st.balloons()


elif choice == '🖊 Comment Section':
    with st.form(key = "myComment", clear_on_submit = True):
        com = st.text_area("Comment your feedback/ suggestions here:",height=100)
        date = st.date_input('Select Date')
    
        if st.form_submit_button("Submit"):

            #st.form_submit_button("Submit")
            comment_section = pd.read_csv('comment_df.csv', index_col=0)
            comment_section = comment_section.append({'Date':str(date), 'Comment': com}, ignore_index=True)
            comment_section.to_csv('comment_df.csv')
            st.success("Your response has been recorded!")
            st.balloons()
            st.write('')

    
