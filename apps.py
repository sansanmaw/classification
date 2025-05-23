import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title='Icecream, Pizza Recognition')
st.title('🍕🍦 Icecream and Pizza Classifier_ by SSM')

@st.cache_resource
def load_trained_model():
    model = keras.models.load_model('pizza_icecream_model.h5', compile=False)
    model.make_predict_function()
    return model

model = load_trained_model()

uploaded_files = st.file_uploader("Upload one or more images", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    with st.spinner('Classifying images...'):
        cols = st.columns(3)
        for i, image_file in enumerate(uploaded_files):
            image = Image.open(image_file)
            image = image.resize((200, 200), Image.Resampling.LANCZOS)
            img_array = np.array(image)
            x = np.expand_dims(img_array, axis=0) / 255.0

            prediction_score = model.predict(x)[0][0]
            prediction = 'Pizza 🍕' if prediction_score > 0.5 else 'Ice Cream 🍦'

            col = cols[i % 3]
            with col:
                st.image(image, use_container_width=True)
                st.markdown(f"**Prediction:** {prediction}")
else:
    st.info("Please upload images to classify.")

if st.button('Clear'):
    st.experimental_rerun()
