import cv2
import numpy as np
import face_recognition

imgElon =face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon , cv2.COLOR_BGR2RGB)
imgtest =face_recognition.load_image_file('imagesBasic/Elon test.jpg')
imgtest = cv2.cvtColor(imgtest , cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[1], faceLoc[2]), (faceLoc[3], faceLoc[0]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (faceLocTest[1], faceLocTest[2]), (faceLocTest[3], faceLocTest[0]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis =face_recognition.face_distance([encodeElon], encodeTest)
print(faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon test', imgtest)
cv2.waitKey(0)