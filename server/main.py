import streamlit as st
import util
import pandas as pd
import numpy as np
import pickle
import json
import numpy as np
import sklearn 

__locations=None
__data_columns=None
__model=None

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./server/assets/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open("./server/assets/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index=__data_columns.index(location.lower())
    except:
        loc_index=-1

    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0: x[loc_index]=1
    return round(__model.predict([x])[0],2)

load_saved_artifacts()

st.write("REAL ESTATE PRICE PREDICTION MODEL")

option=get_location_names()

sqft=st.number_input("Insert the total area of the plot in sqft",min_value=300,key='total_sqft')
bath=st.number_input("Insert the number of bathrooms",min_value=1,key='baths')
bhk=st.number_input("Insert the number of BHK",min_value=1,key='bhks')
location= st.selectbox("Select a Location",options=__locations)

button=st.button("Predict Price")

if button:
    st.write(get_estimated_price(location,sqft,bhk,bath)," Lakhs")
else: st.write("Model Prediction")


