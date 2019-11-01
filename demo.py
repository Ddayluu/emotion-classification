from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_img", type=str, default='demo.jpeg', help='path to image')
    parser.add_argument("-v", "--video_path", type=str, default='test.mp4', help='path to video')
    return parser.parse_args()

# Detect face
def face_detector(img):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Convert image to grayscale
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x,w,y,h))
    return rects, allfaces, img

# Put text
def put_text(rects, faces, image, classifier):
    i = 0
    class_labels = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
    }

    if rects == (0,0,0,0):
        print('No faces detected')

    print("Faces detected: {}".format(len(faces)))
    for face in faces:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        print(preds)
        label = class_labels[preds.argmax()]   
        print(preds.argmax())
        #Overlay our detected emotion on our pic
        label_position = (rects[i][0] , abs(rects[i][2]))
        i =+ 1
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    return image

# Handle each frame as an image
def handle_frame(img, classifier):
    rects, faces, image = face_detector(img)

    try:
        img = put_text(rects, faces, image, classifier)
    except Exception as e:
        raise e

    return img

# Capture video stream
def capture_video_stream(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        # Use webcam
        raise e
        print("No video file detected, use webcam instead")
        cap = cv2.VideoCapture(0)

    return cap

def main(args):
    # Load model
    classifier = load_model('face_and_emotion_detection/emotion_detector_models/model_v6_23.hdf5')

    # Detect as video
    if args.video_path:

        cap = capture_video_stream(args.video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = 10

        out = cv2.VideoWriter('output.avi', 
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            fps,
            (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = handle_frame(frame, classifier)
            out.write(frame)
            # Exit the loop
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break 
    # Detect as image
    else:
        img = cv2.imread(args.file_img)
        img = handle_frame(img, classifier)
        cv2.imwrite('output.jpg', img)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)