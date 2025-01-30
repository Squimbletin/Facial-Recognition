import cv2
import numpy as np
import face_recognition
import os
import threading

path = 'imagesAttendance'
images = []
Names = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    Names.append(os.path.splitext(cl)[0])

print(Names)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("No face detected in image.")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)
frame_skip = 3  # Process every 3rd frame to reduce lag
frame_count = 0
current_encodings = []

def process_frame(frame):
    global current_encodings
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    current_encodings = (encodeCurFrame, facesCurFrame)

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1

    # Only process every `frame_skip` frame
    if frame_count % frame_skip == 0:
        threading.Thread(target=process_frame, args=(img.copy(),)).start()

    if current_encodings:
        encodeCurFrame, facesCurFrame = current_encodings
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = Names[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
