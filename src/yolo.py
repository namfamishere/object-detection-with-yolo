import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
    help='path to input image')
ap.add_argument('-y', '--yolo', required=True, 
    help='base path to YOLO directory')
ap.add_argument('-c', '--confidence', type=float, default=0.5, 
    help='minimum probability to filter weak detections')
ap.add_argument('-t', '--threshold',type=float, default=0.3, 
    help='threshold when applying non-maxima suppression')
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labels_path = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labels_path).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# derive the paths to the YOLO weights and model configuration
weights_path = os.path.sep.join([args['yolo'], "yolov3.weights"])
config_path = os.path.sep.join([args['yolo'], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# load the input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving the bounding boxes and asscociated probabilities
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layer_outputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("YOLO took {:.6f} seconds".format(end-start))

# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
boxes = []  # the bounding boxes around the object
confidences = []  # confidence value that YOLO assigns to an object
class_ids = []  # the detected object's class label

# loop over each of the layer ouputs
for output in layer_outputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e, probability) of the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence >  args["confidence"]:
            # scale the bounding box coordinates back relative to the size of the image, 
            # keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box followed by the boxes'width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (center_x, center_y, width, height) = box.astype('int')

            # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
            x = int(center_x - (width/2))
            y = int(center_y - (height/2))

            # update the list of bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# appy non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y+ w), color, 2)
        text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
        cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


