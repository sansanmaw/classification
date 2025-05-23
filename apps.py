import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras


st.set_page_config(page_title='Icecream, Pizza Recognition')
st.title('Icecream and Pizza Classification')

image_file = st.file_uploader('Choose the Image', ['jpg', 'png'])


@st.cache_resource
def load_trained_model():
    model = keras.models.load_model('pizza_icecream_model.h5', compile=False)
    model.make_predict_function()
    return model


if image_file is not None:
    st.image(image_file, caption='Image', use_column_width=True)
    image = Image.open(image_file)
    image = image.resize((200, 200), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    x = np.expand_dims(img_array, axis=0)/255.0
   
    model = load_trained_model()
    prediction_score = model.predict(x)[0][0]  

    # Interpret prediction
    prediction = 'Icecream' if prediction_score > 0.5 else 'Pizza'
    st.markdown(f"<h3>The image is predicted as: <span style='color:blue'>{prediction}</span></h3>", unsafe_allow_html=True)
