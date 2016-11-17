import sys
import landmarks
import emptionclassifier


def main():
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == 'train':
            if len(sys.argv) >= 6:
                faces = landmarks.training_land_marks(sys.argv[2], sys.argv[3], sys.argv[4])
                training_data, testing_data = emptionclassifier.split_data(faces)
                print 'number of training examples = ' + str(len(training_data))
                print 'number of testing examples  = ' + str(len(testing_data)) + '\n'
                classifier = emptionclassifier.EmotionClassifier(len(training_data[0][1]), sys.argv[5])
                classifier.train(training_data, testing_data, int(sys.argv[6]))
            else:
                print ''

        elif mode == 'classify':
            print 'Do a Thing'

    else:
        print 'Please enter either \'train\' or \'classify\' as command line arguments.'

if __name__ == '__main__':
    main()
