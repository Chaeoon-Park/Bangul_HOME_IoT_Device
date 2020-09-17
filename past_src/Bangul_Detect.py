import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import Dog_Tracking_code
import motor_rotate
import json
import urllib.request
import urllib.parse
import ssl
import math
context = ssl._create_unverified_context()
url = 'https://13.124.126.131'
port = 443
last_time = -1
now_time = 0
dogs = []







def giveserver(DogFlag, angle) :
    uri = url +'/home/state?' #바꿀것
    data = dict()
    data.update({"DOG_FLAG" : DogFlag})
    data.update({"ROTATE_ANGLE" : angle})
    print(data)
    data = urllib.parse.urlencode(data)
    req=urllib.request
    d=req.urlopen(uri + data, context=context)
    result = json.loads(d.read().decode())
    return 
    #얘는 계속 돌리는지랑, 있는지 없는지만 주면 됨.




# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

#서버에서 기기를 실행시키는 명령이 올때까지는 대기해야합니다. 3초에 한번 정도 간격으로 요청을 보내서 디바이스가 켜져있는지 확인 할 수 있도록 합니다.
result = dict()
result['home_running'] = True
result['auto_mode'] = True
result['re_detect'] = False
result['track_dog'] = 1
past_time = time.time()
recommend_past_time = -1
now_angle = 90

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    #프론트 기기가 HOME 기기를 사용 할 때 까지, 요청을 보내며 스핀락 형태로 대기합니다
    while result['home_running'] == False :
        a=1
        #giveserver()
        #time.sleep(3)
        #past_time = time.time()
    while result['auto_mode'] == False :
        a=1
        #오토모드에서는 카메라 회전만 존재합니다.
        #giveserver() 로 받아온 명령에 따라 돌아가는 일만 하세요


    #OpenCV 및 감지 부분
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Grab frame from video stream
    frame1 = videostream.read()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    #코드 부분

    gridbox = []
    measured = []
    cnt = 0
    interval = time.time() - past_time
    past_time = time.time()

    #검출된 객체들중 조건에 맞는 (강아지이며, 검출 점수가 50점 이상) 객체들을 저장합니다.
    for i in range(len(scores)) :
        if ((scores[i] > 0.5) and (scores[i] <= 1.0) and labels[int(classes[i])] == 'person'):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            gridbox.append([ymin,xmin,ymax,xmax])
            measured.append([(xmin+xmax)/2 , ymax])

    #여기서 calculator 처리 해준다
    #정확도를 올리기 위해서 지금 처음 검출된 친구는 따라가지 않도록 한다.
    angles,dogs = Dog_Tracking_code.calculator(dogs,measured,gridbox, interval)
    for i in range(len(dogs)) :
        hotdog = dogs[i]
        ymin = hotdog.ymin 
        xmin = hotdog.xmin 
        ymax = hotdog.ymax
        xmax = hotdog.xmax
        x = hotdog.x
        y = hotdog.y
        object_name = "dog" + str(i+1)
        label = '%s: x: %.2f m , y: %.2f m' % (object_name, x,y) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label textcv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

    if angles[result['track_dog']] != None and angles[result['track_dog']] != 0 :
        now_angle = motor_rotate.camera_motor_control(now_angle,angles[result['track_dog']])
    
    
    # 사용자의 상태를 받아오고, 강아지 좌표값을 반환합니다
    # give_server
    # 재탐색 로직은 서버랑 같이

    
    


    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()