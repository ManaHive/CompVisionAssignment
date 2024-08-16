import os
os.environ["PAFY_BACKEND"] = "internal"

from ultralytics import YOLO
import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import pafy
import yt_dlp as youtube_dl
import tempfile

import settings

def load_model(model_path):
    model = YOLO(model_path)
    return model

def predict_and_plot(model, image, conf):
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Predict using the selected model
    results = model.predict(image_np, conf=conf)

    # Extract boxes and labels from the results
    combined_image = image_np.copy()
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])  # Convert to a regular float
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[class_id]

        # Set color to blue (BGR: (255, 0, 0))
        color = (255, 0, 0)

        # Draw the bounding box and label on the image
        cv2.rectangle(combined_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - label_size[1] if y1 - label_size[1] > 10 else y1 + label_size[1]
        cv2.putText(combined_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the numpy array back to a PIL image
    combined_image = Image.fromarray(combined_image)

    return combined_image

def process_uploaded_video(model, video_file, conf):
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(video_file.read())
    temp_video_file.close()

    vid_cap = cv2.VideoCapture(temp_video_file.name)
    st_frame = st.empty()

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            # Resize the image to a standard size
            image = cv2.resize(image, (720, int(720*(9/16))))
            
            # Predict the objects in the image using the YOLOv8 model
            res = model.predict(image, conf=conf)
               
            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            st_frame.image(res_plotted,
                           caption='Detected Video',
                           channels="BGR",
                           use_column_width=True
                           )
        else:
            vid_cap.release()
            break

    # Remove the temporary video file
    os.remove(temp_video_file.name)

def get_youtube_video_url(youtube_url):
    ydl_opts = {
        'format': 'best',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get('url', None)
    return video_url

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    if st.sidebar.button('Detect Objects'):
        try:
            video_url = get_youtube_video_url(source_youtube)
            vid_cap = cv2.VideoCapture(video_url)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error loading video: {str(e)}")

def _display_detected_frames(conf, model, st_frame, image):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    
    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)
       
    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
