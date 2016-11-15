import os
import cv2
import dlib
import glob
import numpy as np
import tensorflow as tf

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
count = 1
faces, labels, lable = [], [], 0
for folder in os.listdir('Images'):
    if os.path.isdir('Images/'+folder):
        for face in glob.glob(os.path.join('Images/'+folder, '*.tiff')):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(cv2.imread(face, cv2.IMREAD_GRAYSCALE))

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")
            detections = detector(image, 1)
            for k,d in enumerate(detections):
                shape = predictor(image, d)
                landmarks = np.zeros((67, 2))
                for i in range(1,68):
                    landmarks[i-1,0] = shape.part(i).x
                    landmarks[i-1,1] = shape.part(i).y
                faces.append(landmarks)
                labels.append(lable)
            count += 1
    print 'Land Marks for lable ' + str(lable) + ' have been extracted'
    lable+=1
print 'All Land Marks have been extracted'

x = tf.placeholder("float", [None, 136]) # Inputs
y = tf.placeholder("float", [None, 7])  # Classes

weights = {
    'h1': tf.Variable(tf.random_normal([136, 256])),
    'h2': tf.Variable(tf.random_normal([256, 256])),
    'out': tf.Variable(tf.random_normal([256, 7]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([7]))
}

layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
model = tf.matmul(layer2, weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(count/100)
        for i in range(total_batch):
            print 'butts'
            #batchX, batchY = Get the landmarks from the images.
            #_, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})
            #avg_cost += c / total_batch
        if epoch % 1 == 0:
            print "Epoch", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"
    correctPrediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    #print "Accuracy:", accuracy.eval({x: TEST_IMAGES, y: TEST_LABELS})