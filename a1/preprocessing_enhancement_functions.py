import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern

# Function to preprocess a single image
def preprocess_image(path, size):
    img = Image.open(path) # open image
    img = img.resize(size) # resize image
    img_array = np.array(img) # image to numpy array
    img_array = img_array / 255.0
    return img_array

# Function for image enhancement
def enhance_image(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Histogram equalization
    equalized_img = cv2.equalizeHist(gray_img)

    # Noise reduction
    denoised_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    
    return denoised_img

# function for segmenting image
def segment_image(img):
    ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    sobel_x = cv2.Sobel(bin_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(bin_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    return bin_img, sobel_combined

# function for extracting features
def extract_features(bin_img, edges):
    # Feature 1: Local Binary Pattern
    lbp = local_binary_pattern(bin_img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    lbp_hist = lbp_hist.astype("float")

    # # Feature 2: Hu Moments
    moments = cv2.moments(bin_img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Feature 3 and 4: Mean and Standard Deviation of Sobel edges
    mean_val = np.mean(edges)
    std_dev_val = np.std(edges)

    # Combine all features into a single feature vector
    features = np.hstack([lbp_hist, hu_moments, mean_val, std_dev_val])

    return features