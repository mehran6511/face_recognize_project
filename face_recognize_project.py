import cv2
import numpy as np

detector = cv2.FaceDetectorYN.create(
    "model/face/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
"model/face/face_recognition_sface_2021dec.onnx","")

cap = cv2.VideoCapture(0)
list_of_people = {}
l2_similarity_threshold = 0.6

text2 = "No face exite now"
text3 = "Picture Taken"
coordinates1 = (0,-10)
coordinates2 = (50,50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color1 = (0,255,0)
color2 = (255,0,0)
color3 = (0,0,255)
thickness = 1
i=0
while True:
    ret, frame = cap.read()
    frame = frame.copy()
    frame = cv2.flip(frame, 1)
    img1Width = int(frame.shape[1])
    img1Height = int(frame.shape[0])
    img1 = cv2.resize(frame, (img1Width, img1Height))

    detector.setInputSize((img1Width, img1Height))
    faces1 = detector.detect(img1)
    if faces1[1] is None:
        continue
    for idx, face in enumerate(faces1[1]):

        coords = face[:-1].astype(np.int32)
        rectang = cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness=2)
        
        face1_align = recognizer.alignCrop(img1, faces1[1][0])
        face1_feature = recognizer.feature(face1_align)

        for index, face2_feature in list_of_people.items():
            l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
            if (l2_score <= l2_similarity_threshold) and (coords[0]+coords[2] > 350) and (coords[1]+coords[3] > 400):
                cv2.putText(rectang, str(index), (coords[0], coords[1]-10), font, fontScale, color1, thickness, cv2.LINE_AA)

    # save your picture
    if cv2.waitKey(1) & 0xFF == ord("t"):
        if faces1[1] is None:
            cv2.putText(frame, text2, coordinates2, font, fontScale, color2, thickness, cv2.LINE_AA)
        else:
            face1_align = recognizer.alignCrop(img1, faces1[1][0])
            face1_feature = recognizer.feature(face1_align)
            list_of_people[i] = face1_feature
            i = i+1
            cv2.putText(frame, text3, coordinates2, font, fontScale, color3, thickness, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()  