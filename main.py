import cv2
import numpy as np 

net=cv2.dnn.readNet("weight.weights","config.cfg")
classes=[]
with open("coco.names","r") as f:
	classes=[line.strip() for line in f.readlines()]

layers_names= net.getLayerNames()
outputlayers=[layers_names[i[0] -1] for i in net.getUnconnectedOutLayers() ]
colors=np.random.uniform(0,255, size=(len(classes), 3))

#load image
img =cv2.imread("room_ser.jpg")
img=cv2.resize(img, None, fx=0.4, fy=0.4)
height,width,channels=img.shape

#detect
blob=cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)



net.setInput(blob)
outs=net.forward(outputlayers)

#showig iformation
class_ids=[]
confidences=[]
boxes=[]

for out in outs:
	for detection in out:
		scores=detection[5:]
		class_id=np.argmax(scores)
		confidence=scores[class_id]
		if confidence >0.5:
		 #object detexted

		 center_x = int(detection[0] * width)
		 center_y = int(detection[1] * height)
		 w=int(detection[2] * width)
		 h=int(detection[3] * height)
		 #rectangle define
		 x=int(center_x -w /2)
		 y=int(center_y -h /2)


		 boxes.append([x, y, w, h])
		 confidences.append(confidence)
		 class_ids.append(class_id)

#bbb

number_object_detected= len(boxes)
font=cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
	x, y , w , h = boxes[i]
	label=str( classes[class_ids[i]])
	color=colors[i]
	cv2.rectangle(img, (x,y), (x + w, y + h),color, 2)
	cv2.putText(img, label, (x, y + 30), font, 3, color, 1)
	

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()