import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ✅ Load the Saved Data
if not os.path.exists("embeddingss.npy") or not os.path.exists("filenamess.pkl"):
    st.error("❌ Required files missing! Run `app.py` first to generate embeddings.")
    st.stop()

feature_list = np.load("embeddingss.npy")
filenames = pickle.load(open("filenamess.pkl", "rb"))

# ✅ Load MobileNetV2 Model
model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# ✅ Ensure "uploads" directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

st.title("👗 Fashion Recommender System")

# ✅ Function to Save Uploaded File
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ✅ Feature Extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# ✅ Find Similar Images using KNN
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# ✅ Upload & Process Image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    # ✅ Display Uploaded Image
    display_image = Image.open(file_path)
    st.image(display_image, caption="Uploaded Image", use_column_width=True)

    # ✅ Extract Features
    features = feature_extraction(file_path, model)

    # ✅ Find Recommendations
    indices = recommend(features, feature_list)

    # ✅ Display Recommended Images
    st.subheader("🔍 Recommended Images")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(indices[0]):
            col.image(filenames[indices[0][i]], use_column_width=True)
