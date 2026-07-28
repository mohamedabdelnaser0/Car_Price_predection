import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle


# ======================
# Load Model
# ======================
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="mohamed22264/car_price_model",
    filename="car_price_model.pkl"
)




# ======================
# Page Config
# ======================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ======================
# CSS
# ======================

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}


h1{
    color:#1f4e79;
}


.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 15px rgba(0,0,0,0.1);

}


.price{

font-size:40px;
font-weight:bold;
color:#008000;

}


</style>

""", unsafe_allow_html=True)



# ======================
# Title
# ======================

st.title("🚗 Car Price Prediction System by eng:Abdelnaser")

st.write(
"""
AI powered application to estimate used car prices.
"""
)


# ======================
# Sidebar
# ======================


st.sidebar.header("Car Information")


year = st.sidebar.slider(
    "Manufacturing Year",
    1990,
    2026,
    2020
)


km = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)


engine = st.sidebar.number_input(
    "Engine CC",
    min_value=500,
    max_value=8000,
    value=1500
)


power = st.sidebar.number_input(
    "Max Power (bhp)",
    min_value=20,
    max_value=1000,
    value=100
)



# categorical

fuel = st.sidebar.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "CNG",
        "Electric"
    ]
)


transmission = st.sidebar.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)


seller = st.sidebar.selectbox(
    "Seller Type",
    [
        "Dealer",
        "Individual",
        "Trustmark Dealer"
    ]
)



owner = st.sidebar.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth Owner"
    ]
)



make = st.sidebar.selectbox(
    "Brand",
    [
        "Maruti",
        "Hyundai",
        "Honda",
        "Toyota",
        "BMW",
        "Mercedes"
    ]
)



color = st.sidebar.selectbox(
    "Color",
    [
        "White",
        "Black",
        "Silver",
        "Blue",
        "Red"
    ]
)



# ======================
# Prediction Button
# ======================


if st.button("🔮 Predict Price"):


    input_data = pd.DataFrame({

        "Year":[year],

        "Kilometer":[km],

        "Engine":[engine],

        "Max Power":[power],

        "Fuel Type":[fuel],

        "Transmission":[transmission],

        "Seller Type":[seller],

        "Owner":[owner],

        "Make":[make],

        "Color":[color]

    })



    # -------------------
    # Encoding
    # -------------------

    input_data["Transmission"] = (
        input_data["Transmission"]
        .map(
            {
            "Manual":1,
            "Automatic":0
            }
        )
    )



    owner_map={

        "First Owner":1,
        "Second Owner":2,
        "Third Owner":3,
        "Fourth Owner":4

    }


    input_data["Owner"] = (
        input_data["Owner"]
        .map(owner_map)
    )



    input_data = pd.get_dummies(input_data)



    # align columns with training

    for col in model.feature_names_in_:

        if col not in input_data.columns:

            input_data[col]=0



    input_data = input_data[
        model.feature_names_in_
    ]



    # scaling

    input_data_scaled = scaler.transform(
        input_data
    )



    prediction = model.predict(
        input_data_scaled
    )[0]



    # ======================
    # Result
    # ======================


    st.markdown(
    f"""
    <div class="card">

    <h2>Estimated Price</h2>

    <div class="price">

    ${prediction:,.0f}

    </div>

    </div>

    """,
    unsafe_allow_html=True
    )



    st.subheader("Car Details")


    st.dataframe(
        input_data
    )
