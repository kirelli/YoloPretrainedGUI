from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2 as cv
import numpy as np


def yolo_detection():
    #grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkinter.filedialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk
        image = cv.imread(path)
        yolo_image = cv.imread(path)

        height, width, channels = yolo_image.shape
        yolo = cv.dnn.readNetFromDarknet("pretrained_model/yolov4_ders.cfg",
                                          "pretrained_model/yolov4_ders_best.weights")
        classes = ["lezyon"]
        layer_names = yolo.getLayerNames()
        outputlayers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
        print(outputlayers)
        colorRed = (0, 0, 255)
        colorGreen = (0, 255, 0)
        blob = cv.dnn.blobFromImage(yolo_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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

        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                start = (x, y)
                end = (x + w, y + h)
                cv.rectangle(yolo_image, start, end, (0, 255, 0), 4)
                cv.putText(yolo_image, label, (x, y - 5), cv.FONT_HERSHEY_PLAIN, 1, colorRed, 2)
                print((x, y - 20))


        # image scaled to get new dimensions
        scale_percent = 40  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        yolo_image = cv.resize(yolo_image, dim, interpolation=cv.INTER_AREA)
        image = cv.resize(image, dim, interpolation=cv.INTER_AREA)


        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        yolo_image = cv.cvtColor(yolo_image, cv.COLOR_BGR2RGB)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # convert the images to PIL format...
        yolo_image = Image.fromarray(yolo_image)
        image = Image.fromarray(image)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        yolo_image = ImageTk.PhotoImage(yolo_image)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = Label(image=yolo_image)
            panelB.image = yolo_image
            panelB.pack(side="right", padx=10, pady=10)

            # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=yolo_image)
            panelA.image = image
            panelB.image = yolo_image


# initialize the window toolkit along with the two image panels


root = Tk()
panelA = None
panelB = None


# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

btn = Button(root, text="MRI Görüntü Seçimi", command=yolo_detection)
btn.pack(side="bottom", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()




