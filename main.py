import sys
import landmarks
import emotionclassifier


def main():
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == 'train':
            if len(sys.argv) > 6:
                faces = landmarks.training_land_marks(sys.argv[2], sys.argv[3], sys.argv[4])
                training_data, testing_data = emotionclassifier.split_data(faces)
                print 'number of training examples = ' + str(len(training_data))
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'
                classifier = emotionclassifier.EmotionClassifier(len(training_data[0][1]), sys.argv[5])
                classifier.train(training_data, testing_data, int(sys.argv[6]))
            else:
                print 'Please add \'Image Dir\' \'Image File Type\' \'Shape Predictor Path\' ' \
                      '\'Session Save Path\' \'Number of Epochs\''

        elif mode == 'classify':
            if len(sys.argv) > 5:
                face = landmarks.land_marks(sys.argv[2], sys.argv[3])
                classifier = emotionclassifier.EmotionClassifier(int(sys.argv[4]), sys.argv[5])
                classification = classifier.classify(face)
                print sys.argv[2] + ' -> ' + str(classification[1])
            else:
                print 'Please add \'Image Path\' \'Shape Predictor Path\' \'Number of Classes\' \'Session Save Path\''
        else:
            'Please add either \'train\' or \'classify\' as command line arguments.'
    else:
        print 'Please add either \'train\' or \'classify\' as command line arguments.'

if __name__ == '__main__':
    main()
