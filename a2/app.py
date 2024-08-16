# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Load the models
pretrained_model, custom_model = None, None

try:
    pretrained_model = helper.load_model(settings.PRETRAINED_MODEL)
except Exception as ex:
    st.error(f"Unable to load pre-trained model. Check the specified path.")
    st.error(ex)

try:
    custom_model = helper.load_model(settings.CUSTOM_MODEL)
except Exception as ex:
    st.error(f"Unable to load custom-trained model. Check the specified path.")
    st.error(ex)

if pretrained_model is None or custom_model is None:
    st.error("Model loading failed. Please check the paths in the settings.")

# If image or video is selected
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)
source_img = None
source_vid = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)
    with col1:
        if source_img is not None:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if source_img is not None:
            if st.sidebar.button('Detect Objects'):
                if custom_model:
                    combined_image = helper.predict_and_plot(custom_model, uploaded_image, confidence)
                    st.image(combined_image, caption='Detected Image', use_column_width=True)
                else:
                    st.error("Custom-trained model needs to be loaded for image detection.")
        else:
            st.warning("Please upload an image for detection.")

elif source_radio == settings.VIDEO:
    source_vid = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "mov", "avi", "mkv"))
    
    if source_vid is not None:
        st.video(source_vid)
        
        if st.sidebar.button('Detect Video Objects'):
            if pretrained_model:
                helper.process_uploaded_video(pretrained_model, source_vid, confidence)
            else:
                st.error("Pre-trained model is not loaded.")
    else:
        st.warning("Please upload a video for detection.")

elif source_radio == settings.YOUTUBE:
    if pretrained_model:
        helper.play_youtube_video(confidence, pretrained_model)
    else:
        st.error("Pre-trained model is not loaded.")

else:
    st.error("Please select a valid source type!")
