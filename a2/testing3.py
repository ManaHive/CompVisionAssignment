from ultralytics import YOLO
import torch
from roboflow import Roboflow

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
    rf = Roboflow(api_key="bqIer3bFPcwTc8qlTwou")
    project = rf.workspace("imageprocessing-1awq9").project("electronic-wastes")
    version = project.version(2)
    dataset = version.download("yolov8")

    # Load the YOLOv8 model - you can choose 'yolov8n.pt', 'yolov8s.pt', etc.
    model = YOLO('yolov8n.pt') 

    # Train the model
    model.train(
        data="C://Users//User//ImgProcessing//Assignment2//Electronic-wastes-2//data.yaml", 
        epochs=100,   
        imgsz=640, 
        batch=16, 
        name="yolov8_electronic_waste"
    )

# Load the trained model
model = YOLO('C://Users//User//ImgProcessing//Assignment2//runs//detect//yolov8_electronic_waste9//weights//best.pt')

# Run inference on a new image
results = model('C://Users//User//ImgProcessing//CompVisionAssignment//images//train//syringe2//1622055562-4802.jpg')

# Display the results
for result in results:
    result.show()


if __name__ == '__main__':
    main()
