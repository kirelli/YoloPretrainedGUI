
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("yolo pretrained image/images/1.jpg")
# print(img)

plt.imshow(img)
plt.show()

height, width, channels = img.shape


yolo = cv2.dnn.readNetFromDarknet("pretrained_model/yolov4_ders.cfg","pretrained_model/yolov4_ders_best.weights")

classes = ["lezyon"]


layer_names = yolo.getLayerNames()

outputlayers = [layer_names[i -1] for i in yolo.getUnconnectedOutLayers()]

print(outputlayers)

colorRed   = (0, 0, 255)
colorGreen = (0, 255, 0)


blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

yolo.setInput(blob)

outputs = yolo.forward(outputlayers)

class_ids = []
confidences = []
boxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if (confidence > 0.5):
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)



indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))


for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label =str(classes[class_ids[i]])
        start = (x,y)
        end = (x+w,y+h)
        cv2.rectangle(img, start, end, (0,255,0), 4)
        cv2.putText(img, label, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, colorRed, 4)
        print((x,y-20))


plt.imshow(img)
plt.show()









