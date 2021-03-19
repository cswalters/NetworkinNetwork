import numpy
import imageio
import pickle
import time
import cProfile
import gc
from sklearn.utils import shuffle

from neuralNetwork import NeuralNetwork


CIFAR_ALL = "/Users/chris/Documents/University/2018/Sem 2/Neural Networks/Datasets/CIFARData.nosync/cifar-10-batches-py/"
CIFAR_LABELS = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


def main(performSnellan=False):

    outputSize = len(CIFAR_LABELS.keys())

    # Learning parameters
    lr = 0.0002
    l2Reg = 8e-6
    learningRateDecay = numpy.float32(996e-3)

    # Network architecture

    """
    Functional convolutional network for CIFAR 10 - 81% val acc
    
    lr = 0.0001
    l2Reg = 8e-6
    learningRateDecay = numpy.float32(96e-2)
    
    layers = [
        {'type': 'conv', 'k': 32},
        {'type': 'conv', 'k': 32}},
        {'type': 'pool', 'method': 'max'},
        {'type': 'conv', 'k': 64},
        {'type': 'conv', 'k': 64},
        {'type': 'pool', 'method': 'max'},
        {'type': 'conv', 'k': 128},
        {'type': 'conv', 'k': 128, 'dropout': 0.3},
        {'type': 'pool', 'method': 'max'},
        #{'type': 'neural', 'k': 256, 'dropout': 0.5},
        {'type': 'output', 'k': outputSize}
    ]
    """

    layers = [
        {'type': 'mlp', 'k': 96, 'updateMethod': 'adam'},
        {'type': 'pool', 'method': 'max'},
        {'type': 'mlp', 'k': 96, 'updateMethod': 'adam', 'dropout': 0.5},
        {'type': 'pool', 'method': 'max'},
        {'type': 'mlp', 'k': 96, 'updateMethod': 'adam'},
        {'type': 'pool', 'method': 'max'},
        {'type': 'output', 'k': outputSize, 'updateMethod': 'adam'}
    ]

    # Retrieve images from folder
    trainIm, trainLab, testIm, testLab = extractCIFARBetter(numBatches=1)

    # Process the image sby subtracting the mean image values and normalising them by batch standard deviation.
    trainIm, trainMean, trainStd = preprocessImages(trainIm, generateValues=True)
    testIm = preprocessImages(testIm, mean=trainMean, std=trainStd, generateValues=False)

    trainIm, trainLab = shuffle(trainIm, trainLab)

    # Initialise the Messiah
    sonOfMan = NeuralNetwork(trainIm.shape[1:], layers, lr, l2Reg=l2Reg)

    # Training hyperparameters
    numEpochs = 10
    batchSize = 1

    print("His story has begun on the back of {0} older stories.".format(len(trainIm)))
    n = 0
    last = time.time()

    if performSnellan:

        eye = numpy.zeros((3, 32, 32))
        eye.fill(1)

        eyePatch = numpy.zeros(trainIm.shape[1:])
        eyePatch[:eye.shape[0],:eye.shape[1], :eye.shape[2]] = eye

        for duration in range(1, 4):
            for onset in range(1, 4):

                for i in range(numEpochs * len(trainIm) + 1):
                    start = i * batchSize % len(trainIm)
                    end = start + batchSize

                    n += 1 * batchSize
                    if n % (len(trainIm) / 10) == 0:
                        print('Another {0} in {1}'.format(n, (time.time() - last)))
                        last = time.time()

                    if start == 0 and i != 0:
                        n = 0

                        sonOfMan.epochCount += 1
                        sonOfMan.lr *= learningRateDecay
                        trainIm, trainLab = shuffle(trainIm, trainLab)
                        testIm, testLab = shuffle(testIm, testLab)

                        print('{0} epoch finished. Learning rate is {1}'.format(sonOfMan.epochCount, sonOfMan.lr))

                        trainloss, trainacc = sonOfMan.predict(trainIm[:1000], trainLab[:1000])
                        print("Training accuracy: {0}".format(trainacc))
                        print("Training loss: {0}".format(trainloss))

                        testloss, testacc = sonOfMan.predict(testIm[:1000], testLab[:1000])
                        print("Testing accuracy: {0}".format(testacc))
                        print("Testing loss: {0}\n".format(testloss))

                        if i == numEpochs * len(trainIm):
                            with open("Results.txt", 'a') as results:
                                results.write("Blinded at Epoch {0} for {1} \
                                epochs resulted in {2}% accuracy.".format(onset, duration, testacc))

                        last = time.time()

                    sonOfMan.t += 1

                    # Perform blind if needed.
                    if i in range(onset * len(trainIm), (onset + duration) * len(trainIm)):
                        blindIm = trainIm[start:end] * eyePatch
                        loss, acc = sonOfMan.epoch(blindIm, trainLab[start:end])
                    else:
                        loss, acc = sonOfMan.epoch(trainIm[start:end], trainLab[start:end])


    else:
        # Run normally.
        for i in range(numEpochs * len(trainIm) + 1):
            start = i * batchSize % len(trainIm)
            end = start + batchSize

            n += 1 * batchSize
            if n % (len(trainIm)/10) == 0:
                print('Another {0} in {1}'.format(n, (time.time() - last)))
                last = time.time()

            if start == 0 and i != 0:

                n = 0

                sonOfMan.epochCount += 1
                sonOfMan.lr *= learningRateDecay
                trainIm, trainLab = shuffle(trainIm, trainLab)
                testIm, testLab = shuffle(testIm, testLab)

                print('{0} epoch finished. Learning rate is {1}'.format(sonOfMan.epochCount, sonOfMan.lr))

                loss, acc = sonOfMan.predict(trainIm[:1000], trainLab[:1000])
                print("Training accuracy: {0}".format(acc))
                print("Training loss: {0}".format(loss))

                loss, acc = sonOfMan.predict(testIm[:1000], testLab[:1000])
                print("Testing accuracy: {0}".format(acc))
                print("Testing loss: {0}\n".format(loss))

                last = time.time()

            sonOfMan.t += 1
            loss, acc = sonOfMan.epoch(trainIm[start:end], trainLab[start:end])

def preprocessImages(oldImages, mean=0, std=0, generateValues=True):

    # First duplicate the images horizontally to replicate bifocal vision.
    images = numpy.tile(oldImages, 2)

    # See CS231n for more info.
    # Process images by mean subtraction
    if generateValues:
        mean = numpy.mean(images, axis=0)

    images -= mean

    # Normalise images by standard deviation
    if generateValues:
        std= numpy.std(images, axis=0)

    images /= std

    images = images.astype(numpy.float32)

    if generateValues:
        return images, mean, std
    else:
        return images

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extractCIFARBetter(numBatches=5):

    # Collate the desired training batches
    train_images = None
    train_labels = []

    for i in range(1, numBatches + 1):
        data = unpickle(CIFAR_ALL + 'data_batch_' + str(i))
        if train_images is None:
            train_images = data[b'data']
        else:
            train_images = numpy.vstack((train_images, data[b'data']))

        train_labels += data[b'labels']

    train_images = train_images.reshape(-1, 3, 32, 32)
    train_images = train_images.astype(numpy.float64)

    # Extract the single testing batch
    data = unpickle(CIFAR_ALL + 'test_batch')
    test_images = data[b'data'].reshape(-1, 3, 32, 32)
    test_images = test_images.astype(numpy.float64)

    test_labels = data[b'labels']

    return train_images, train_labels, test_images, test_labels

def extractor(filepath):

    with open(filepath, 'r') as file:
        content = file.readlines()

    for line in content

if __name__ == '__main__':
    cProfile.run("main(True)", sort="time")