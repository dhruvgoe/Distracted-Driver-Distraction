import streamlit as st
import tensorflow as tf
import joblib
from PIL import Image, ImageOps
import numpy as np
import cv2
import sys

sys.path.append('src')

# Load Models
try:
    cnn_model = tf.keras.models.load_model('model_files/cnn_model.h5')
    pca_model = joblib.load('model_files/pca_model.joblib')
    svm_model = joblib.load('model_files/svm_model.joblib')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def preprocess_image(image, use_cnn=True):
    if use_cnn:
        size = (100, 100)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(img, (100, 100))
        flattened_img = resized_img.flatten()
        return flattened_img.reshape(1, -1)

st.title("Distracted Driver Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # CNN Prediction
        cnn_input = preprocess_image(image, use_cnn=True)
        cnn_prediction = cnn_model.predict(cnn_input)

        if(cnn_prediction>=10):
            cnn_prediction = cnn_prediction % 10

        # Corrected handling of CNN prediction
        cnn_predicted_class_index = np.argmax(cnn_prediction)  # Get index of highest probability
        cnn_confidence = cnn_prediction[0][cnn_predicted_class_index] #Get the confidence score

        cnn_class_names = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']

        if len(cnn_class_names) != cnn_prediction.shape[1]:
            st.error(f"CNN class names ({len(cnn_class_names)}) do not match prediction output size ({cnn_prediction.shape[1]}). Check your model and class names.")
            st.stop()

        cnn_predicted_class_name = cnn_class_names[cnn_predicted_class_index]

        st.subheader("CNN Prediction")
        st.write(f"Prediction: {cnn_predicted_class_name}")
        st.write(f"Confidence: {cnn_confidence:.2f}")

        # PCA + SVM Prediction
        pca_svm_input = preprocess_image(image, use_cnn=False)
        pca_input = pca_model.transform(pca_svm_input)
        svm_prediction = svm_model.predict(pca_input)

        pca_svm_class_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

        if len(pca_svm_class_names) != svm_prediction.shape[0]:
            st.error(f"PCA+SVM class names ({len(pca_svm_class_names)}) do not match prediction output size ({svm_prediction.shape[0]}). Check your model and class names.")
            st.stop()

        svm_predicted_class_name = pca_svm_class_names[svm_prediction[0]]

        st.subheader("PCA + SVM Prediction")
        st.write(f"Prediction: {svm_predicted_class_name}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")