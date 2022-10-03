import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 

st.set_option('deprecation.showPyplotGlobalUse', False)


data = pd.read_csv("data//Life_Expectancy_Data_updated.csv")
x = np.array(data[' BMI ']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Life expectancy ']))

st.title("Life Expectancy Calculation")
st.image("data//img.jpeg",width = 650)
nav = st.sidebar.radio("Navigation",["Home","Prediction","About"])

if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(data)
    
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    val = st.slider("Filter data using BMI",2,77)
    data = data.loc[data[" BMI "]>= val]
    if graph == "Non-Interactive":
        #fig = plt.figure(figsize = (12, 7))
        plt.figure(figsize = (10,5))
        plt.scatter(data[" BMI "],data["Life expectancy "])
        plt.ylim(0)
        plt.xlabel("Body Mass Index")
        plt.ylabel("Life Expectancy")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout =go.Layout(
            xaxis = dict(range=[0,100]),
            yaxis = dict(range =[0,100])
        )
        fig = go.Figure(data=go.Scatter(x=data[" BMI "], y=data["Life expectancy "], mode='markers'),layout = layout)
        st.plotly_chart(fig)
if nav == "Prediction":
    st.header("Get your Life Expectanacy")
    val = st.number_input("Enter you BMI",2.00,77.10,step = 0.25)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your Life Expectanacy is {round(pred)} Years")
if nav == "About":
    st.header("About")
    st.text("Life Expectancy is an analytical as well as a statistical measure of the longevity\nof the population depending upon distinct factors. Over the years, Life expectancy\nobservations are being vastly used in medical, healthcare planning, and pension-related\nservices, by concerned government authorities and private bodies. Advancements\nin forecasting, predictive analysis techniques, and datascience technologies have\nnow made it possible to develop accurate predictive models. In many countries, it is\na matter of political debate about how to decide the retirement age and how to manage\nthe financial issues related to the public matter. Life expectancy predictions provide\nsolutions related to these issues in many developed countries. With the advancement\nin new systematic, accurate, efficient, and result-oriented techniques in the field\nof Data Science, now predictions of the Life Expectancy of the selected region are\nbecoming more prominent in demand of the government authorities and the private bodies\nand their policymaking.")