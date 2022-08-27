import numpy as np
import cv2
yolo= cv2.dnn.readNet("yolov3-tiny_6000.weights", "yolov3-tiny.cfg" )
classes= []
with open("classes.names",'r') as f:
  classes= f.read().splitlines()
x1= "  Number of truck="
x2= "  Number of car="
x3= "  Number of bike="
x4= "  Total Vehicle="
k=0
OverLay = ["green-","red-","yellow-"]
status=[[1,0,0],[0,1,0],[0,0,1]]
cap=cv2.VideoCapture("tt.mp4")
while True: 
  _, img = cap.read()
  height, width, _ = img.shape
  blob= cv2.dnn.blobFromImage(img,1.0/255, (608,608),(0,0,0),swapRB=True,crop=False)
  yolo.setInput(blob)
  output_layers_name = yolo.getUnconnectedOutLayersNames()
  layeroutput = yolo.forward(output_layers_name)
  boxes=[]
  confidences=[]
  class_ids=[]
  
  NumBikeL = 0
  NumCarL = 0
  NumTruckL = 0
  NumBikeR = 0
  NumCarR = 0
  NumTruckR = 0
  countL=0
  countR=0

  for output in layeroutput:
    for detection in output:
      score=detection[5:]
    
      class_id=np.argmax(score)
    
      confidence=score[class_id]
      if confidence>0.5:
        center_x= int(detection[0]*width)
        center_y= int(detection[1]*height)
        w= int(detection[2]*width)
        h= int(detection[3]*height)
        x=int(center_x-w/2)
        y=int(center_y-h/2)
        if True:
        #if (2*y-3*x+1340>=0 and y>200 and x+y-600>0 ):
        

          boxes.append([x,y,w,h])
          confidences.append(float(confidence))
          class_ids.append(class_id)
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  font = cv2.FONT_HERSHEY_PLAIN
  colors = np.random.uniform(0, 255, size = (len(boxes), 3))
  if len(indexes) > 0:
    for i in indexes.flatten():
      if class_ids[i] == 0 or class_ids[i] == 1 or class_ids[i] == 2:
        x,y,w,h = boxes[i]
        if y>220:
          if y-4*x+3000>0: 
            countL+=1     
            if class_ids[i] == 0:
              NumCarL += 1
            if class_ids[i] == 1:
              NumBikeL += 1
            if class_ids[i] == 2:
              NumTruckL += 1
          if y-4*x+3000<0:
            countR+=1
            if class_ids[i] == 0:
              NumCarR += 1
            if class_ids[i] == 1:
              NumBikeR += 1
            if class_ids[i] == 2:
              NumTruckR += 1

          label = str(classes[class_ids[i]])
          confi = str(round(confidences[i], 2))
          color = colors[i]

          cv2.rectangle(img , (x,y), (x+w, y+h), color, 1)
          cv2.putText(img, label +"  "+confi, (x, y+20), font, 1, (255,255,0), 1)
    img = cv2.putText(img, x1 + str(NumTruckL), (5,20),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x2+str(NumCarL) , (5,40),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x3 +str(NumBikeL), (5,60),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x4 +str(countL), (5,80),font, 1, (102, 255, 0), 2)

    img = cv2.putText(img, x1 + str(NumTruckR), (800,40),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x2+str(NumCarR) , (800,40),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x3 +str(NumBikeR), (800,60),font, 1, (102, 255, 0), 2)
    img = cv2.putText(img, x4 +str(countR), (800,80),font, 1, (102, 255, 0), 2)
    #img = cv2.line(img, (780,120), (940,775), (255,0,0), 2)
    l=k%3
    m=(k+1)%3
    cv2.rectangle(img , (390,20), (830, 160), (0,0,0), 2)
    img = cv2.putText(img, "Traffic Light Status " ,(400,60),font, 2.5, (255,20,100),2)
    for j in range(len(OverLay)):
      img = cv2.putText(img, OverLay[j]+str(status[l][j]),(450,100+20*j),font, 2, (0,0,255),2)
      img = cv2.putText(img, OverLay[j]+str(status[m][j]),(650,100+20*j),font, 2, (0,0,255),2)
    cv2.imshow('Image',img)
    k=k+1
    key= cv2.waitKey(1)
    if key==27:
      break
  cap.release()
  cv2.destroyAllWindows()
