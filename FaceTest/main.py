import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

original = cv2.imread('FaceTest/Jacob.jpg')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))

detections = detector(image, 1)
for k,d in enumerate(detections):
    shape = predictor(image, d)
    for i in range(1,68):
        print 'x = ', shape.part(i).x, ' y = ', shape.part(i).y
        cv2.circle(original, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)

cv2.imshow("Image", original)

cv2.waitKey(0)
