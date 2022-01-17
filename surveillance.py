import cv2
import csv23 as csv
import collections
import numpy as np
from tracker import *
import math

weightsPath='C:\\Users\\workplace\\yolov4-tiny.weights'
configPath='C:\\Users\\workplace\\yolov4-tiny.cfg'
videoPath='C:\\Users\\workplace\\surveillance-clip.mp4'

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def ad(a, b):
    return a+b

# Initialize Tracker
tracker = EuclideanDistTracker()

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.5
font_color = (0,255,0)
font_size = 0.5
font_thickness = 1

# center line position
center_line_position = 320

# The list classes
classes = []
with open('coco.names', 'r') as f:
    classes=[line.strip() for line in f.readlines()]
print(classes)

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]
detected_classes = []

# configure the network model
net = cv2.dnn.readNet(configPath, weightsPath)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicles
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy==center_line_position-15):
        up_list[index]=up_list[index]+1
        
    elif (iy==center_line_position+15):
        down_list[index]=down_list[index]+1
        
    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classes 
    height, width = img.shape[:2]
    boxes = []
    class_ids = []
    confidences = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in required_class_index:
                if confidence > confThreshold:
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        # print(x,y,w,h)
        color = [int(c) for c in colors[class_ids[i]]]
        name = classes[class_ids[i]]
        detected_classes.append(name)
        # Draw classes and confidence
        cv2.putText(img,f'{name.upper()} {int(confidences[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(class_ids[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

def video_stream(videoPath):
    cap=cv2.VideoCapture(videoPath)
    input_size=1280*720
    while True:
        ret, img = cap.read()
        print(img,ret)
        imgr = cv2.resize(img,(1080, 640),None,fx=None,fy=None)
        ih, iw, _ = imgr.shape
        blob = cv2.dnn.blobFromImage(imgr, 1 / 255, (640,480), (0, 0, 0), True, False)
       
        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        postProcess(outputs,imgr)

        # Draw the crossing lines
        cv2.line(imgr, (0, center_line_position), (iw, center_line_position), (255, 0, 0), 1)

        # Draw counting texts in the frame
        cv2.putText(imgr, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(imgr, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(imgr, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(imgr, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(imgr, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(imgr, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100),cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # The original input frame is shown in the window 
        
        # Show the frames
        if ret:
            cv2.imshow('Output', imgr)
            cv2.waitKey(1)
        else:
            break
    
    # Write the object counting information in a file and save it
    with open("data.csv", 'w') as f:
        cwriter = csv.writer(f)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'Truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f.close()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_stream(videoPath)