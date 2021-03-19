import numpy

from neuron import Neuron
from layer import Layer
import util


class NeuralLayer(Layer):

    def __init__(self, inputSize, k, updateMethod='adam', activationMethod='relu', dropout=1):

        # Array to hold the neurons of the layer.
        self.neurons = []
        # Holds the latest output of this layer.
        self.forwardResult = None
        # Number of neurons
        self.k = k
        # Whether to use dropout and which dropout.
        # Currently, not using dropout.
        self.dropout = dropout

        self.updateMethod = updateMethod
        self.activationMethod = activationMethod

        # Checks if a volume is being inputted into layer.
        if isinstance(inputSize, tuple):
            inputSize = numpy.product(inputSize)

        # Generate neurons of the layer.
        for n in range(k):
            n = Neuron(inputSize)

            # If using Adam update, initialise each neuron with an Adam mass and velocity.
            if updateMethod == 'adam':
                n.m, n.v = 0, 0
            elif updateMethod == 'rms':
                n.cache, n.v = 0, 0

            self.neurons.append(n)

    def predict(self, batch):

        # Check if batch inputs are given as volumes instead of vectors. Flatten if so.
        if batch.ndim > 2:
            batch = batch.reshape(batch.shape[0], -1).T

        forwardResult = []

        # Determine activations of each neuron.
        for n in self.neurons:

            # Apply activation function if using.
            if self.useActivation:
                if self.activationMethod == 'relu':
                    forwardResult.append(util.relu(n.strength(batch)))
                elif self.activationMethod == 'sigmoid':
                    forwardResult.append(util.sigmoid(n.strength(batch)))

            else:
                forwardResult.append(n.strength(batch))

        # Flatten list of numpy.arrays into one numpy array and store record as layers more recent output.
        self.forwardResult = numpy.array(forwardResult)

        return self.forwardResult

    def forward(self, batch):

        # Check if batch inputs are given as volumes instead of vectors. Flatten if so.
        if batch.ndim > 2:
            batch = batch.reshape(batch.shape[0], -1).T

        forwardResult = []

        # Initialise l2 parameter
        l2 = 0

        for n in self.neurons:

            # Apply activation function if using.
            if self.useActivation:
                if self.activationMethod == 'relu':
                    forwardResult.append(util.relu(n.strength(batch)))
                elif self.activationMethod == 'sigmoid':
                    forwardResult.append(util.sigmoid(n.strength(batch)))

            else:
                forwardResult.append(n.strength(batch))

            # Record throughput of neuron for backprop/update purposes.
            n.latestInput = batch

            # Update l2 for each neurons regularisation value.
            l2 += n.regularisation()

        # Flatten list of numpy.arrays into one numpy array and store record as layers more recent output.
        self.forwardResult = numpy.array(forwardResult)

        return self.forwardResult, l2

    def backward(self, d, needNextDelta=True):

        # Check if delta is given as a volume instead of a vector. Flatten if so.
        if d.ndim > 2:
            d = d.reshape(d.shape[1], -1)

        weights = []

        # Deactivate the error gradients of this layer's output (d).
        if self.useActivation:
            if self.activationMethod == 'relu':
                delta = d * util.dRelu(self.forwardResult)
            elif self.activationMethod == 'sigmoid':
                delta = d * util.dSigmoid(self.forwardResult)

        else:
            delta = d

        # Assign error gradients to the individual neurons.
        for index, n in  enumerate(self.neurons):
            n.delta = delta[index]

            # Store each neurons weights for backpassing the delta.
            if needNextDelta:
                weights.append(n.weights)

        # If not calculating delta of the prev layer, end here.
        if not needNextDelta:
            return
        else:
            # Else, flatten weights array into single numpy array.
            weights = numpy.array(weights)
            # Backpass the delta using transposed weight vectors/matrices.
            return weights.T.dot(delta)

    def outputSize(self):
        if self.forwardResult:
            # Return the dimensions of the layer's most recent output, save the number of examples
            # in the batch: batch.shape[0]\
            return self.forwardResult.shape[1:]
        else:
            # If no output as of yet, return the number of neurons in the layer.
            return self.k

    def update(self, lr, l2Reg, t=0):
        if self.updateMethod == 'adam':
            util.adamUpdate(self.neurons, lr, t=t, l2Reg=l2Reg)
        elif self.updateMethod == 'rms':
            util.RMSPropUpdate(self.neurons, lr, l2Reg=l2Reg)
        elif self.updateMethod == 'vanilla':
            util.vanillaUpdate(self.neurons, lr, l2Reg=l2Reg)
