import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title("Iris Flower Species Prediction")
st.markdown("This app predicts the species of iris flowers based on their features.")   

st.header("Input Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal Length (cm) ")
    sepal_length = st.slider('sepal_length', 4.0, 8.0, 5.0)
    sepal_width = st.slider('sepal_width', 2.0, 5.0, 3.0)

with col2:
    st.text("Petal Length (cm) ")
    petal_length = st.slider('petal_length', 1.0, 7.0, 4.0)
    petal_width = st.slider('petal_width', 0.1, 2.5, 1.5)

st.text('')

if st.button('predict'):
    result = predict(np.array([sepal_length, sepal_width, petal_length, petal_width]))
    predicted_class = result[0]
    st.success(f'The predicted species is: {predicted_class}')

    #display image
    image_path = f"images/{predicted_class.lower()}.jpg"
    st.image(image_path, caption=f"Image of {predicted_class}", use_container_width=True)
