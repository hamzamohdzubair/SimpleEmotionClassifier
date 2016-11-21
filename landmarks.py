import os
import cv2
import dlib
import glob
import numpy as np
from dlib import rectangle


def training_land_marks(dir, file_type, shape_predictor):
    """ Gets the Landmarks and labels of facial images stored in folders according to their classification.
    :param dir: The directory storing the image folders.
    :type dir: A file path to a folder ending with a forward slash e.g. '~/Images/'.
    :param file_type: The file type of the images to be read.
    :type file_type: A image file type e.g. 'tiff'.
    :param shape_predictor: A file path to the shape predictor used by the dlib library.
    The file can be found here http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    :type shape_predictor: A file path.
    :return: A list of tuples each containing a list of x and y values of facial landmarks and a classification label.
    :rtype: A list of tuples each containing a list and a int.
    """
    faces, label, count = [], 0, 0
    for folder in sorted(os.listdir(dir)):
        if os.path.isdir(dir + folder):
            for face in glob.glob(os.path.join(dir + folder, '*.' + file_type )):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                image = clahe.apply(cv2.imread(face, cv2.IMREAD_GRAYSCALE))
                detector, predictor = dlib.get_frontal_face_detector(), dlib.shape_predictor(shape_predictor)
                detections = detector(image, 1)
                for _,d in enumerate(detections):
                    left, right, top, bottom = d.left()-20, d.right()+20, d.top()-20, d.bottom()+20
                    image = image[top:bottom, left:right]
                    image = cv2.resize(image, (200, 200))
                detections = detector(image, 1)
                for k,d in enumerate(detections):
                    shape = predictor(image, d)
                    landmarks = []
                    labels = np.zeros(7)
                    for i in range(1,68):
                        landmarks.append(shape.part(i).x)
                        landmarks.append(shape.part(i).y)
                        labels[label] = 1
                    faces.append((landmarks,labels))
                count += 1
            print 'Land Marks for label ' + str(label) + ' have been extracted'
            label+=1
    print 'All Land Marks have been extracted from ' + dir + '\n'
    return faces


def land_marks(image_file, shape_predictor):
    """ Gets the landmarks of a image file.
    :param image: A file path to a image file.
    :type image: A file path.
    :param shape_predictor: A file path to the shape predictor used by the dlib library.
    The file can be found here http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    :type shape_predictor: A file path.
    :return: A list of x and y values of facial landmarks.
    :rtype: A list.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))


    detector, predictor = dlib.get_frontal_face_detector(), dlib.shape_predictor(shape_predictor)
    landmarks, detections = [], detector(image, 1)
    face = np.zeros((1,134))
    for _, d in enumerate(detections):
        left, right, top, bottom = d.left() - 25, d.right() + 25, d.top() - 25, d.bottom() + 25
        image = image[top:bottom, left:right]
        image = cv2.resize(image, (200, 200))
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        for i in range(1, 68):
            landmarks.append(shape.part(i).x)
            landmarks.append(shape.part(i).y)
    face[0] = landmarks
    print 'Land Marks for ' + image_file + ' have been extracted'
    return face
