import numpy
import chainer.functions

from neuralLayer import NeuralLayer
import util


class ConvLayer(NeuralLayer):

    def __init__(self, inputSize, k, f=3, s=1, p=1, updateMethod='adam', activationMethod='relu', dropout=1):

        # Record dimensions of input to this layer.
        self.imageSize = 0
        self.depth = inputSize[0]
        self.height = inputSize[1]
        self.width = inputSize[2]

        # Number of filters
        self.k = k
        # Filter dimension (as a square)
        self.f = f
        # Stride of convolution
        self.s = s
        # Padding of input
        self.p = p

        # Calculate output dimensions.
        self.depth2 = k
        self.height2 = (self.height - self.f + 2 * self.p) // self.s + 1
        self.width2 = (self.width - self.f + 2 * self.p) // self.s + 1

        # Create neuralLayer that has input size as size of a filter for im2col matrix multiplication.
        super().__init__(f*f*self.depth, k, updateMethod=updateMethod, activationMethod=activationMethod, dropout=dropout)

    def predict(self, batch):

        # Store number of examples in batch as image size.
        self.imageSize = batch.shape[0]

        # Convert image volume to flattened 2D matrix of (Num_Filter_Locations_in_Image x Flattened Filter)
        cols = util.im2col_indices(batch, self.f, self.f, self.p, self.s)

        sumWeights = []
        bias = []

        # Populate weight and bias lists
        for n in self.neurons:
            bias.append(n.bias)
            sumWeights.append(n.weights)

        # Flatten sumWeights to single numpy array.
        sumWeights = numpy.array(sumWeights)

        # Determine activation (or strength) of layer. Multiple col by the weights, add the biases, reshaped to a 2D matriix.
        # Reshape produced strength back into a 3D volume and then transpose to have the example number as first dimension.
        strength = (sumWeights.dot(cols) + numpy.array(bias).reshape(sumWeights.shape[0], 1))
        strength = strength.reshape(self.k, self.height2, self.width2, -1).transpose(3, 0, 1, 2)

        # Activate the strength if necessary
        if self.useActivation:
            if self.activationMethod == 'relu':
                return util.relu(strength)
            elif self.activationMethod == 'sigmoid':
                return util.sigmoid(strength)

        else:
            return strength

    def forward(self, batch):

        # Store number of examples in batch as image size.
        self.imageSize = batch.shape[0]

        # Convert image volume to flattened 2D matrix of (Num_Filter_Locations_in_Image x Flattened Filter)
        #cols = util.im2col_indices(batch, self.f, self.f, self.p, self.s)

        cols = chainer.functions.im2col(batch, self.f, stride=self.s, pad=self.p).array
        nInput = cols.reshape(cols.shape[0], cols.shape[1], -1)
        cols = nInput.transpose(1, 2, 0).reshape(cols.shape[1], -1)

        l2 = 0
        sumWeights = []
        bias = []

        # Populate weight and bias lists as with predict but also record the input to the neuron
        # and sum regularisation scores.
        for n in self.neurons:
            # n.latestInput = cols
            n.latestInput = nInput

            bias.append(n.bias)
            sumWeights.append(n.weights)
            l2 += n.regularisation()

        # Flatten sumWeights to single numpy array.
        sumWeights = numpy.array(sumWeights)

        # Determine activation (or strength) of layer. Multiple col by the weights, add the biases, reshaped to a 2D matrix.
        # Reshape produced strength back into a 3D volume and then transpose to have the example number as first dimension.
        strength = (sumWeights.dot(cols) + numpy.array(bias).reshape(sumWeights.shape[0], 1))
        strength = strength.reshape(self.k, self.height2, self.width2, -1).transpose(3, 0, 1, 2)

        # Activate the strength if necessary.
        # Same as with the predict() function except store record of the output of this layer.
        if self.useActivation:
            if self.activationMethod == 'relu':
                self.forwardResult = util.relu(strength)
            elif self.activationMethod == 'sigmoid':
                self.forwardResult = util.sigmoid(strength)

        else:
            self.forwardResult = strength

        return self.forwardResult, l2


    def backward(self, d, needNextDelta=True):

        # If delta is not a volume, reshape and transpose it such that it is a volume of output size
        # and number of images in batch.
        if d.ndim < 4:
            d = d.reshape(self.width2, self.height2, self.k, -1).T

        # Deactivate
        if self.useActivation:
            if self.activationMethod == 'relu':
                delta = d * util.dRelu(self.forwardResult)
            elif self.activationMethod == 'sigmoid':
                delta = d * util.dSigmoid(self.forwardResult)
        else:
            delta = d

        sumWeights = []

        # Distribute delta's to each neuron. Index corresponds to the filter number index, where each
        # filter is its own neuron. Flatten the delta into a vector the length of the flattened filter.
        for index, n in enumerate(self.neurons):
            #n.delta = delta[:, index, :, :].transpose(1, 2, 0).flatten()
            n.delta = delta[:, index, :, :].reshape(delta.shape[0], -1).transpose(1, 0)
            if needNextDelta:
                # Reshape filter/neuron weights from vector to volume of weights and then rotate the filter.
                rot = numpy.rot90(n.weights.reshape(self.depth, self.f, self.f), k=2, axes=(1, 2))
                sumWeights.append(rot)

        if not needNextDelta:
            return

        # Calculate number of paddings
        padding = ((self.width - 1) * self.s + self.f - self.width2) // 2
        # Reshape delta into the flat matrix (the col) for back convolution.
        cols = util.im2col_indices(delta, self.f, self.f, padding=padding, stride=self.s)

        # Transpose weights once more and then flatten into matrix of (num_filters x flattened_filter)
        sumWeights = numpy.array(sumWeights).transpose(1, 0, 2, 3).reshape(self.depth, -1)

        # Back convolve.
        result = sumWeights.dot(cols)

        # Reproduce 3D volume.
        im = result.reshape(self.depth, self.height, self.width, -1).transpose(3, 0, 1, 2)

        return im

    def outputSize(self):
        return (self.depth2, self.height2, self.width2)

    def update(self, lr, l2Reg, t=0):
        # Use same update methods as with basic neural layer.
        super().update(lr, l2Reg=l2Reg, t=t)