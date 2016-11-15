import os
import cv2
import dlib
import glob
import numpy as np

"""
from adiencealign.pipeline.CascadeFaceAligner import CascadeFaceAligner
cascade_face_aligner = CascadeFaceAligner(haar_file="resources/haarcascade_frontalface_default.xml",
                                          lbp_file='resources/lbpcascade_frontalface.xml',
                                          fidu_model_file = 'resources/model_ang_0.txt',
                                          fidu_exec_dir = 'resources/')

cascade_face_aligner.detect_faces("Images/", "Faces/")

cascade_face_aligner.align_faces(input_images = "Faces/",
                                         output_path = "AlignedFaces/",
                                         fidu_max_size = 200*200,
                                         fidu_min_size = 50*50,
                                         is_align = True,
                                         is_draw_fidu = True,
                                         delete_no_fidu = True)
"""
faces, labels, c, lable = [], [], 0, 0
for folder in os.listdir('Images'):
    if os.path.isdir('Images/'+folder):
        for face in glob.glob(os.path.join('Images/'+folder, '*.tiff')):
            #print face + ' with lable ' + str(lable)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(cv2.imread(face, cv2.IMREAD_GRAYSCALE))

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
            detections = detector(image, 1)
            for k,d in enumerate(detections):
                shape = predictor(image, d)
                landmarks = np.zeros((68, 2))
                for i in range(1,68):
                    landmarks[i-1,0] = shape.part(i).x
                    landmarks[i-1,1] = shape.part(i).y
                faces.append(landmarks)
                labels.append(lable)
        lable+=1
#print 'Done'

for i in range(len(faces)):
    print faces[i]
    print 'lable=' + str(labels[i])
    print '\n\n'