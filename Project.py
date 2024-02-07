import tkinter as tk
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()

score = 0
rpred = [99]
lpred = [99]
capture_video = False


def detect_blink():
    global score, rpred, lpred, capture_video

    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Define the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            if rpred[0] == 1:
                lbl = 'Open'
            if rpred[0] == 0:
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            if lpred[0] == 1:
                lbl = 'Open'

            if lpred[0] == 0:
                lbl = 'Closed'
            break

        if rpred[0] == 0 and lpred[0] == 0:
            score = score + 1
        else:
            score = score - 1

        if score < 0:
            score = 0

        score_text = f'Score: {score}'
        cv2.putText(frame, lbl, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, score_text, (10, height - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 15:
            # person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass

        cv2.imshow('frame', frame)

        if capture_video:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoWriter object
    out.release()

    cap.release()
    cv2.destroyAllWindows()


def toggle_video_capture():
    global capture_video
    capture_video = not capture_video
    if capture_video:
        capture_button.config(text='Stop Capture')
    else:
        capture_button.config(text='Start Capture')


# Create the GUI window
window = tk.Tk()
window.title('Drowsiness Detection')
window.geometry('400x250')

# Create a button to start/stop the eye blink detection
start_button = tk.Button(window, text='Start Detection', command=detect_blink)
start_button.pack(pady=10)

# Create a button to start/stop video capture
capture_button = tk.Button(window, text='Start Capture', command=toggle_video_capture)
capture_button.pack(pady=10)

window.mainloop()
