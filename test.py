from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda,merge, Dense, Flatten, MaxPooling2D 
from keras.layers.merge import Add
from keras.regularizers import l2
from keras import backend
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy

import numpy as np
import numpy.random 
import os
import dill as pickle
import matplotlib.pyplot as plot
%matplotlib inline
from sklearn.utils import shuffle
from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces()


def initialWeights(shape, name=None):
    #Initialization as in paper
    values = numpy.random.normal(loc=0, scale=0.01, size=shape)
    return backend.variable(values, name=name)

def initialBiases(shape, name=None):
    #Initialization as in paper
    values = numpy.random.normal(loc=0.5, scale=0.01, size=shape)
    return backend.variable(values, name)


imageSize = 64
def buildNet():
    #Olivetti faces are 64 x 64 pixels
    inputShape = (imageSize, imageSize, 1)
    leftInput  = Input(inputShape)
    rightInput = Input(inputShape)
    convolutionNet = Sequential()
    #Testing was done with much smaller layers, for speed. 
    #The same network is used for both sides of the siamese network, here initialized with convolution layers
    convolutionNet.add(Conv2D(64, (9, 9), activation='relu', input_shape = inputShape, kernel_initializer = initialWeights, bias_initializer = initialBiases ) )
    convolutionNet.add(MaxPooling2D(2))
    convolutionNet.add(Conv2D(128, (7, 7), activation='relu', input_shape = inputShape, kernel_initializer = initialWeights, bias_initializer = initialBiases ) )
    convolutionNet.add(MaxPooling2D(2))
    convolutionNet.add(Conv2D(128, (4, 4), activation='relu', input_shape = inputShape, kernel_initializer = initialWeights, bias_initializer = initialBiases ) )
    convolutionNet.add(MaxPooling2D(2))
    convolutionNet.add(Conv2D(256, (3, 3), activation='relu', input_shape = inputShape, kernel_initializer = initialWeights, bias_initializer = initialBiases ) )
    convolutionNet.add(Flatten())
    #Dense combination before comparison
    convolutionNet.add(Dense(4096, activation="sigmoid", kernel_regularizer=l2(0.001), kernel_initializer= initialWeights, bias_initializer= initialBiases))
    
    #Convolutional net is applied to both inputs
    leftResult = convolutionNet(leftInput)
    rightResult= convolutionNet(rightInput)
    
    #Then the difference is taken. Merge is a deprecated layer, but I could not figure out the non-deprecated way of doing this
    L1 = lambda x: backend.abs(x[0] - x[1])
    first = lambda x: x[0]
    distance = merge([leftResult, rightResult], mode=L1, output_shape = first )
    #The distance is then used to predict the probability of the images being of the same class
    prediction = Dense(1, activation="sigmoid", bias_initializer = initialBiases, kernel_initializer= initialWeights) (distance)
    siameseNet = Model(input=[leftInput, rightInput], output = prediction)
    
    optimizer = SGD()
    siameseNet.compile(loss="binary_crossentropy", optimizer = optimizer)
    siameseNet.count_params()
    return siameseNet

siameseNet = buildNet()
#Full network has 5,000,000 free parameters. The test network has about 14,000
print(siameseNet.count_params())

#Curently no hyper-parameter optimization, learning decay, or affine distortions


#There are 400 images, with 10 images in each class. We will take 24 classes (240 images, 60% of data)
#to use to build our network. 60% is taken from the paper.
numberOfClasses = 40
numberTrainingClasses = 24
numberTestClasses = numberOfClasses - numberTrainingClasses
classSize = 10
trainingData = data.images[:numberTrainingClasses*classSize]
trainingLabels = data.target[:numberTrainingClasses*classSize]
testData = data.images[numberTrainingClasses*classSize:]
testLabels =data.target[numberTrainingClasses*classSize:]

#Generates a batch of tests, half from the same class, half from different classes. 
def generateBatch(size):
    pairs = [np.zeros((size, imageSize, imageSize, 1)), np.zeros((size, imageSize,imageSize, 1))]
    targets= np.zeros(size)
    for i in range(size):
        category1 = numpy.random.randint(numberTrainingClasses)
        item1 = numpy.random.randint ( classSize)
        #if even element, choose another element from same category. If odd, pick from other category
        if ( i % 2 == 0):
            category2 = category1
            item2 = (item1 + numpy.random.randint (1, classSize)) % classSize
            targets[i] = 1
        else:
            category2 = (category1 + numpy.random.randint(1, numberTrainingClasses)) % numberTrainingClasses
            item2 = numpy.random.randint ( classSize)
        pairs[0][i, :, :, :] = trainingData[category1*classSize + item1].reshape(imageSize, imageSize, 1)
        pairs[1][i, :, :, :] = trainingData[category2*classSize + item2].reshape(imageSize, imageSize, 1)
    return pairs, targets
    
#Generates one example from each test class to compare against
def generateOneShotSamples():
    classExamples = numpy.zeros((numberTestClasses, imageSize, imageSize, 1))
    indices = numpy.zeros(numberTestClasses)
    for i in range(numberTestClasses):
        indices[i] = numpy.random.randint(classSize)
        classExamples[i, :, :, :] = testData[i*classSize + indices[i]].reshape(imageSize, imageSize, 1)
    return (indices, classExamples)

#Generates examples from test set that are not the excluded indices, so that 
#we do not try to categorize an item as itself
def generateOneShotTests(size, indicesExclude):
    tests = numpy.zeros((size, imageSize, imageSize, 1))
    labels = numpy.zeros(size)
    for i in range(size):
        index = numpy.random.randint(len(testData))
        answer = testLabels[index] 
        while(index % classSize == indicesExclude[answer- numberTrainingClasses]):
            index = numpy.random.randint(len(testData))
            answer = testLabels[index]
        tests[i,:,:,:] = testData[index].reshape(imageSize, imageSize, 1)
        labels[i] = answer
    return (tests, labels)
    
#How often to check on new classes and how large to make each check
checkFrequency = 100
checkSize = 2* numberTestClasses 

#Force into the form that Keras expects. No real math being done here
def makePairs(test, examples):
    
    currentTestPairs = []
    for j in range (len(examples)):
        currentTestPairs.append([numpy.array([tests[i]]), numpy.array([examples[j]])])
    return (currentTestPairs)


#Actually run the code, for a thousand iterations
loss = numpy.zeros(1000)
testError = []
#First for loop for output test
for i in range(1):
#for i in range(1000):
    (inputs, targets) = generateBatch(24)
    loss[i] = siameseNet.train_on_batch(inputs, targets)
    print("iteration " + str(i) + " with loss " + str(loss[i])+ "\n")
    #If it is time to check on the test set, do so. 
    if ( i % checkFrequency == 0):
        indices, examples = generateOneShotSamples()
        tests, labels = generateOneShotTests(checkSize, indices)
        numCorrect = 0
        for j in range(checkSize):
            #This is 
            print("Testing on iteration " + str(i) + " and trial " + str(j) + "\n")
            testPairs = makePairs(tests[j], examples)
            probabilities = numpy.zeros(numberTestClasses)
            for k in range (numberTestClasses):
                probabilities[k] = siameseNet.predict(testPairs[k], numberTestClasses)
            if (np.argmax(probabilities) == labels[j]):
                numCorrect+=1
        accuracy = numCorrect*100 / checkSize
        testError.append(accuracy)
        print("Testing on iteration " + str(i) + " with accuracy " + str(accuracy) + "%\n")
plt.figure()
plt.plot(loss)
plt.show()
plt.plot(testError)
plt.show()