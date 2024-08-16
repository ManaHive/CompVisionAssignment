import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import joblib
import import_ipynb
import preprocessing_enhancement_functions as pre_en
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

model = joblib.load("knn_model.pkl")

st.set_page_config(
    page_title = "Skin Cancer Classifier Applet: Benign or Malignant Tumor?",
    page_icon = "🏻",
    layout="centered"
)

def predict_image(img):
    img_arr = pre_en.preprocess_image(img, (500,500))
    enhanced_img = pre_en.enhance_image((img_arr * 255).astype('uint8'))
    bin_img, edge = pre_en.segment_image(enhanced_img)
    input_features = pre_en.extract_features(bin_img, edge)

    columns = [f'lbp_{i}' for i in range(9)]  # LBP feature names
    columns += [f'hu_moment{i}' for i in range(7)] # Hu Moments
    columns += ['mean_edge', 'std_dev_edge']  # Mean and std dev names

    input_df = pd.DataFrame([input_features], columns=columns)
    input_df.drop(columns=['lbp_2','lbp_4','lbp_6'], inplace=True)

    return model.predict(input_df)

st.title("Skin Lesion Classifier Applet: Benign or Malignant Tumor?")
st.write("Here, you can upload images of skin lesions, and using a pretrained classfier model trained on a skin imaging dataset, the applet will segment the image, and predict whether the uploaded image shows a benign or malignant tumor.")
st.caption("Note: The classifer model used for this applet is K-Nearest Neighbours, and after hyperparameter tuning, achieved a final accuracy rating of 79.40%.") 
st.caption("Please do not use this applet for actual medical diagnosis, it can only make predictions, and cannot be fully relied upon.")

st.divider()
col1, col2 = st.columns([1.5,1], gap="large")
result = None
with col1:
    uploaded_file = st.file_uploader("Please insert an image here:", type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        result = predict_image(uploaded_file)

with col2:
    st.subheader("Model Evaluation")
    st.divider()
    if result[0] == "benign":
        st.write("The model predicts that this skin lesion is a **benign** tumor.")
    elif result[0] == "malignant":
        st.write("The model predicts that this skin lesion is a **malignant** tumor.")
