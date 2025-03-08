import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image  

# Load the trained model
model = load_model('C:\\Users\\preet\\Documents\\visual studio code\\ML_Fruits&Veg\\Image_classify.keras')

# Class labels
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
            'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
            'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
            'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
            'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
            'tomato', 'turnip', 'watermelon']

img_width, img_height = 180, 180

# Ensure previous results are cleared when a new image is uploaded
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None
    st.session_state.confidence = None

# Streamlit UI
st.header('🥦🍎 Fruits & Vegetables Image Classification Model')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    st.image(original_image, caption="Uploaded Image", use_container_width=True) 

    model_image = tf.keras.utils.load_img(uploaded_file, target_size=(img_width, img_height))
    img_array = tf.keras.utils.img_to_array(model_image)
    img_batch = tf.expand_dims(img_array, axis=0)   

with st.spinner("🔍 Analyzing the image..."):
    prediction = model.predict(img_batch)
    score = tf.nn.softmax(prediction[0]).numpy()


# Display result
st.write(f'The Fruit or the Vegetable is  {data_cat[np.argmax(score)]}')
st.write(f'Accuracy:  {np.max(score) * 100:.2f}%')
