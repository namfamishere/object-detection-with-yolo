# Intro
YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep CNN to detect an object.
In this work, we will use the weights of the pre-trained YOLOv3 model that was pre-trained on the COCO dataset with 80 classes to build the object detector in images and videos.
# Requirements
## Packages:
- Numpy
- OpenCV
## The pre-trained YOLOv3:
You need the labels file `coco.names`, the cfg file `yolov3.cfg` and the weights file `yolov3.weights` of the pre-trained YOLOv3.

The labels and cfg file are available for you in the directory `yolo-coco/`, you only need download the weights file [here](https://pjreddie.com/media/files/yolov3.weights) or directly download using terminal and place it in the same directory as them:
`$ wget https://pjreddie.com/media/files/yolov3.weights`
# How to run
## Object detection in images
Check for the options: `python yolo_image.py --help`

Example for object detection on the input image: `airport.jpg` and outputing the result image `airport_res.jpg`:

`python yolo_image.py --input inputs/airport.jpg --output outputs/airport_res.jpg --yolo yolo-coco`

Sample output image:
![](https://raw.githubusercontent.com/namfamishere/object-detection-with-yolo/main/outputs/airport_res.jpg)

## Object detection in videos

Check for the options: `python yolo_video.py --help`

Example for object detection on the input video:
`python yolo_video.py --input inputs/street-walk-in-Hanoi.mp4 --output outputs/street-walk-in-Hanoi_res.avi --yolo yolo-coco`

View sample ouput video [here](https://drive.google.com/file/d/1i2pJT-SKyT7DE1VfsI2f8qfSwM6c6F1U/view?usp=sharing)
