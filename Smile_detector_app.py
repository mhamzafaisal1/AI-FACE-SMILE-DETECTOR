import cv2
from random import randrange

from numpy import eye

# Load some pre-trained data on face frontals from open cv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# Choosing the video/webcam to detect faces in, now using video capture it will look for videos and if 0 is passed to it them it defaults to a webcam.
webcam = cv2.VideoCapture(0)


# iterate forever over frames until the video ends
while True:

    # read the current frame
    successful_frame_read, frame = webcam.read()

    # if there is an error abort
    if not successful_frame_read:
        break
    # Now make the image black and white for the algorithm to understand
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # drawing the rectangles around the face to easily identify the faces
    # looping through the data if there are more faces
    for (x, y, w, h) in face_coordinates:
        # face_coordinates is an array itself so [0] is to select the first one or the first face
        # (x, y, w, h) = face_coordinates[0]
        # the last two parts are for the color (Blue-Green-Red, thickness of the rectangle)
        cv2.rectangle(frame,  (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # rand range is a function being used to make sure different colors pop up every time for the face detection

        # GET THE SUB-FRAME (USING NUMPY N-DIMENSIONAL ARRAY SLICING)
        the_face = frame[y:y+h, x:x+w]

        # change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # minNeighbors is used because smiles are hard to distinguish if there are more constant smiles in one area then the chances of the smile being real is high so the last two arguments are used. And scale factor means to blur the image but not too much where is unrecognizable but enough where only the prominent features are clear
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # eyes
        eyes = eye_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.1, minNeighbors=5)

        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face,  (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)

        for (x_, y_, w_, h_) in eyes:
            cv2.rectangle(the_face,  (x_, y_), (x_+w_, y_+h_), (0, 0, 0), 2)

        # label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow("Smile Detector", frame)

    # In python this command is used to keep the window open until a key is pressed to clear it otherwise the window quickly shows up and closes, it is hard to notice.
    # using the variable key to store what key is being pressed and then using it later on
    key = cv2.waitKey(1)

    # to quit the app by using the key "Q", the key is being fetched from the waitKey function and the ASCII characters are being compared
    if key == 81 or key == 113:
        break


# to release the videocapture object
webcam.release()
cv2.destroyAllWindows()
