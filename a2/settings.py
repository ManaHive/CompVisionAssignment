import os
from pathlib import Path

# Get the absolute path of the current file
file_path = Path(__file__).resolve()
root_path = file_path.parent

# Default Project Root
ROOT = Path(os.getenv('PROJECT_ROOT', root_path.relative_to(Path.cwd())))

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
YOUTUBE = 'YouTube'
SOURCES_LIST = [IMAGE, VIDEO, YOUTUBE]

# Images config (default, can be overridden by environment variables)
DEFAULT_IMAGES_DIR = Path(os.getenv('IMAGES_DIR', "C:/User/ImgProcessing/CompVisionAssignment/mobile_phone"))
DEFAULT_IMAGE = DEFAULT_IMAGES_DIR / os.getenv('DEFAULT_IMAGE', '3335_066136652881514101_720.jpg')
DEFAULT_DETECT_IMAGE = DEFAULT_IMAGES_DIR / os.getenv('DEFAULT_DETECT_IMAGE', 'your_detected_image.jpg')

# Videos config (default, can be overridden by environment variables)
DEFAULT_VIDEO_DIR = Path(os.getenv('VIDEO_DIR', ROOT / 'videos'))

# Directly referencing the specific video in Downloads
SPECIFIC_VIDEO_PATH = Path(os.getenv('SPECIFIC_VIDEO', r"C:\Users\User\Downloads\WhatsApp Video 2023-03-18 at 1.43.01 AM.mp4"))

# Dictionary to easily reference videos
VIDEOS_DICT = {
    'specific_video': SPECIFIC_VIDEO_PATH,
}

# ML Model config (default, can be overridden by environment variables)
MODEL_DIR = Path(os.getenv('MODEL_DIR', ROOT / 'weights'))
PRETRAINED_MODEL = MODEL_DIR / Path(os.getenv('PRETRAINED_MODEL', 'yolov8n.pt'))
CUSTOM_MODEL = Path(r"C:\Users\User\ImgProcessing\Assignment2\runs\detect\yolov8_electronic_waste9\weights\best.pt")
SEGMENTATION_MODEL = MODEL_DIR / os.getenv('SEGMENTATION_MODEL', 'yolov8n-seg.pt')
