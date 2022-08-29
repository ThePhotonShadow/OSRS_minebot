import cv2
import win32api, win32con
import argparse
import numpy as np
from mss import mss
from PIL import Image
import time

#Global status flag
itera = 1

#A nice click function is helpful later
def click(x,y):
    for i in range(20):
        for j in range(20):
            win32api.SetCursorPos((x - 20 + i,y - 20 + j))
    win32api.SetCursorPos((x,y))
    time.sleep(0.010)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(0.005)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    
#These do exactly what they say on the sticker
def shiftdown():
    win32api.keybd_event(160, 0, 1, 0)
    
def shiftup():
    win32api.keybd_event(160, 0, 2, 0)
    

#Network init stuff
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default='C:/Users/pnkid/Desktop/images/newbuilt/custom/tinyexp.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default='C:/Users/pnkid/Desktop/images/newbuilt/backup/tiny_57000.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default='C:/Users/pnkid/Desktop/images/newbuilt/custom/objects.names')
args = ap.parse_args()


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Drawing boxes for testing CV and pretty output for powerpoint
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
# Pretty window for powerpoint and demo
window_title= "OSRS Detection"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)


# Get ore names
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Colorpicking
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)

# VideoCap
cap = cv2.VideoCapture(0)

mon = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}

sct = mss()

# Core Bot logic
while cv2.waitKey(1) < 0:
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imshow('test', image)
    hasframe, image2 = cap.read()
    #image=cv2.resize(image, (620, 480)) 
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    #print(len(outs))
    
    # 2 output(outs) from 2 different scales [3 bounding box per each scale]
    
    # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
    # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
    # and the second output will be = 2028x6=26x26x18 (18=3*6) 
    
    for out in outs: 
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Select a priority box
    leftmost = None
    leftval = 999999
    for i in indices:
        if boxes[i[0]][0] < leftval:
            leftmost = i[0]
    
    itera -= 1
    
    # I found I needed lots of artificial delay here to even get clicks to register, I suspect the Java input detection is low-tick
    if itera <= 0 and leftmost != None:
        xl = int(boxes[leftmost][0]) + int(boxes[leftmost][2] / 4)
        xll = xl + int(boxes[leftmost][2]) / 3
        yt = int(boxes[leftmost][1]) + int(boxes[leftmost][3]) / 4
        yb = yt + int(boxes[leftmost][3]) / 3
        #Click a target ore, twice for good detection measure
        for flip in range(2):
            click(int(boxes[leftmost][0] + boxes[leftmost][2]/2 + np.random.randint(-8, 8)), int(boxes[leftmost][1] + boxes[leftmost][3]/2 + np.random.randint(-8, 8)))
            time.sleep(0.007)
        if np.random.randint(3) > 0:
            #Drop items so inventory does not overflow
            time.sleep(.020)
            shiftdown()
            click(2394, 1225)
            shiftup()
            shiftdown()
            click(2391, 1260)
            shiftup()
            shiftdown()
            click(2439, 1226)
            shiftup()
        time.sleep(.015)
        win32api.SetCursorPos((int(boxes[leftmost][0] + boxes[leftmost][2]/2 + np.random.randint(-8, 8)),int(boxes[leftmost][1] + boxes[leftmost][3]/2 + np.random.randint(-8, 8))))
        # I don't have logic for box persistence, so I have no way of figuring out if a box in the previous frame is the same as a box now, so I have to guess at when to start mining something else
        # I sample chisquare for delay with a transform for this because I want to look like a human player, but I'd rather not miss much ore
        itera = int(50 + 2*(np.random.chisquare(2)))
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   
    # Speed readout
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0))
    
    cv2.imshow(window_title, image)