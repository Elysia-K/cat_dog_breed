import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# 모델과 라벨 경로
MODEL_PATH = 'data/petbreed_model'
LABEL_PATH = 'data/petbreed_labels.csv'
IMG_SIZE = (299, 299)

# 모델 로드
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# 라벨 로드
@st.cache_data
def load_labels():
    df = pd.read_csv(LABEL_PATH)
    return df['label'].tolist()

# 예측 함수
def predict_breed(model, class_names, img: Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    return class_names[pred_idx], preds[0][pred_idx]

# Streamlit 앱
st.title("🐶🐱 Cat and Dog Breed Classifier")
st.write("Upload an image and we'll guess what breed it is!")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded image', use_column_width=True)

    with st.spinner('Predicting...'):
        model = load_model()
        class_names = load_labels()
        label, confidence = predict_breed(model, class_names, image)

    st.success(f"✅ prediction result: **{label}** ({confidence:.2%} certainty)")
