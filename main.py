import cv2

video = 'assets/vids/cats.mp4'
# def detect_cats(video_path):
cat_cascade = cv2.CascadeClassifier('assets/xml/cats.xml')

video = cv2.VideoCapture(video)
print("""Simple cats face detector using these assets:
cv2 library,
XML file from OpenCV GitHub: https://github.com/opencv/opencv/tree/master/data/haarcascades,
and this YouTube video: https://www.youtube.com/watch?v=-5CdAup0o-I

https://www.linkedin.com/in/ondrejpacovsky/
https://github.com/Ondra9071""")

while True:
    ret, frame = video.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cats = cat_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('linkedin.com/in/ondrejpacovsky', frame)

video.release()
cv2.destroyAllWindows()