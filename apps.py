import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title='Icecream, Pizza Recognition')
st.title('San San Maw_🍦 Icecream and 🍕 Pizza Classification')

# Sidebar options
threshold = st.sidebar.slider('Prediction threshold', 0.0, 1.0, 0.5)
show_confidence = st.sidebar.checkbox('Show confidence score', True)

# Multiple image upload
image_files = st.file_uploader('Choose images', type=['jpg', 'png'], accept_multiple_files=True)

@st.cache_resource
def load_trained_model():
    model = keras.models.load_model('pizza_icecream_model.h5', compile=False)
    model.make_predict_function()
    return model

if image_files:
    model = load_trained_model()
    
    for image_file in image_files:
        image = Image.open(image_file).resize((200, 200), Image.Resampling.LANCZOS)
        st.image(image, caption=image_file.name, use_container_width=True)

        img_array = np.array(image)
        x = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner('Predicting...'):
            prediction_score = model.predict(x)[0][0]

        prediction = 'Pizza' if prediction_score > threshold else 'Icecream'
        st.markdown(
            f"<h3>The image is predicted as: <span style='color:blue'>{prediction}</span></h3>",
            unsafe_allow_html=True
        )

        if show_confidence:
            st.write(f"Confidence score: {prediction_score:.2f}")

else:
    st.info('Please upload one or more images to get started!')
