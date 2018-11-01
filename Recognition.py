import tkinter as tk                # python 3
from tkinter import font as tkfont,Canvas, Label, Tk, StringVar, Button, LEFT,messagebox
import cv2, math
import numpy as np
from os import listdir
from os.path import isfile, join


def regfacetraining():
    
    data_path = 'C:/Users/Ranajoy/Downloads/OpenCV/Faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('C:/Users/Ranajoy/AppData/Local/Programs/Python/Python36-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break

    cap.release()
    return cv2.destroyAllWindows()

def regface():
        
    face_classifier = cv2.CascadeClassifier('C:/Users/Ranajoy/AppData/Local/Programs/Python/Python36-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


    def face_extractor(img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return None

        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face


    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'C:/Users/Ranajoy/Downloads/OpenCV/Faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100:
            break

    cap.release()
    print('Colleting Samples Complete!!!')

    return cv2.destroyAllWindows()


def reg():
    # Open Camera
    capture = cv2.VideoCapture(0)

    while capture.isOpened():

        # Capture frames from the camera
        ret, frame = capture.read()

        # Get hand data from the rectangle sub window
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        crop_image = frame[100:300, 100:300]

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

        # Change color-space from BGR -> HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

        # Kernel for morphological transformation
        kernel = np.ones((5, 5))

        # Apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Apply Gaussian Blur and Threshold
        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        # Show threshold image
        cv2.imshow("Thresholded", thresh)

        # Find contours
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            # Find contour with maximum area
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            # Create bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Find convex hull
            hull = cv2.convexHull(contour)

            # Draw contour
            drawing = np.zeros(crop_image.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
            count_defects = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                # if angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

                cv2.line(crop_image, start, end, [0, 255, 0], 2)

            # Print number of fingers
            if count_defects == 0:
                cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
            elif count_defects == 1:
                cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 2:
                cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 3:
                cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            elif count_defects == 4:
                cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            else:
                pass
        except:
            pass

        # Show required images
        cv2.imshow("Gesture", frame)
        all_image = np.hstack((drawing, crop_image))
        cv2.imshow('Contours', all_image)

        # Close the camera if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    return cv2.destroyAllWindows()


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="    Fun With Cam :)    ", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="reg", command=reg)
        button2 = tk.Button(self, text="regface", command=regface)
        button3 = tk.Button(self, text="regfacetraining", command=regfacetraining)

        button1.pack()
        button2.pack()
        button3.pack()
        label = tk.Label(self, text="", font=controller.title_font)
        label.pack(side="top", fill="x", pady=2)

        
class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
