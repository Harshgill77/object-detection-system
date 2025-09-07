# Object Detection System

A Python-based object detection system using OpenCV and YOLOv4 for real-time detection on local images or video.

## Files
- project.py → main detection script
- yolov4.cfg → YOLOv4 configuration
- coco.names → class labels

> Note: yolov4.weights is NOT included due to size. Download it here: [YOLOv4 weights](https://pjreddie.com/media/files/yolov4.weights)

## Usage
1. Clone the repo:
```bash
git clone git@github.com:Harshgill77/object-detection-system.git
cd object-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run detection:
```bash
python project.py
```

## License
MIT License
