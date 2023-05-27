# Cats detection using OpenCV library.
# https://github.com/Ondra9071/catsdetector/

import cv2

def detect_cats(video_path):
    xml = cv2.CascadeClassifier('assets/xml/cats.xml')

    source = cv2.VideoCapture(video_path)

    while True:
        ret, frame = source.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cats = xml.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in cats:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('https://github.com/Ondra9071/catsdetector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    source.release()
    cv2.destroyAllWindows()

video_path = 'assets/vids/cats.mp4'
detect_cats(video_path)
