import numpy
import torch

import util
from layer import Layer
from neuralLayer import NeuralLayer


class MLPLayer(Layer):

    def __init__(self, inputSize, k=192, f=3, s=1, p=1, updateMethod='adam', activationMethod='relu', dropout=1):

        self.layers = []

        self.networks = []
        self.forwardResult = None
        self.latestInput = None

        self.updateMethod = updateMethod
        self.activationMethod = activationMethod
        self.dropout = dropout

        self.imageSize = 0
        self.depth = inputSize[0]
        self.height = inputSize[1]
        self.width = inputSize[2]

        # Number of filters
        self.k = k
        # Filter dimension (as a square)
        self.f = f
        # Stride of MLP
        self.s = s
        # Padding of input
        self.p = p

        # Calculate output dimensions.
        self.depth2 = k
        self.height2 = (self.height - self.f + 2 * self.p) // self.s + 1
        self.width2 = (self.width - self.f + 2 * self.p) // self.s + 1

        # Perceptron architecture. Defaulting to 3 layer perceptron used in the Network in Network paper.
        self.filterArch = [
            {'type': 'neural', 'k': k, 'updateMethod': updateMethod, 'activationMethod': activationMethod},
            {'type': 'neural', 'k': k, 'updateMethod': updateMethod, 'activationMethod': activationMethod},
            {'type': 'output', 'k': k, 'updateMethod': updateMethod, 'activationMethod': activationMethod}
        ]

        nextInputSize = self.f*self.f*self.depth

        for layer in self.filterArch:
            # Generate the appropriate layers for network.
            if layer['type'] == 'neural':
                layer.pop('type')
                neural = NeuralLayer(nextInputSize, **layer)
                self.layers.append(neural)
                nextInputSize = neural.outputSize()

        return

    def predict(self, batch):

        # Store number of examples in batch as image size.
        self.imageSize = batch.shape[0]

        # Convert image volume to flattened 2D matrix of (Flattened Filter X Num_Filter_Locations)
        nextInput = util.im2col_indices(batch, self.f, self.f, self.p, self.s)

        for index, layer in enumerate(self.layers):
            nextInput = layer.predict(nextInput)

        result = numpy.array(nextInput)
        result = result.reshape(self.k, self.height2, self.width2, -1).transpose(3, 0, 1, 2)

        return result

    def forward(self, batch):

        # Store number of examples in batch as image size.
        self.imageSize = batch.shape[0]
        self.latestInput = batch.shape

        # Convert image volume to flattened 2D matrix of (Flattened Filter X Num_Filter_Locations)
        nextInput = util.im2col_indices(batch, self.f, self.f, self.p, self.s)

        l2 = 0

        for index, layer in enumerate(self.layers):

            # Get tuple of result and l2 value
            layerResult = layer.forward(nextInput)
            # Store result for next layer to process
            nextInput = layerResult[0]
            # Update regularisation
            l2 += layerResult[1]

        result = numpy.array(nextInput)
        #print(result.shape)
        result = result.reshape(self.k, self.height2, self.width2, -1).transpose(3, 0, 1, 2)

        return result, l2

    def backward(self, d, needNextDelta=True):

        #### Backpropagation ####
        if d.ndim < 4:
            d = d.reshape(self.width2, self.height2, self.k, -1).T

        backInput = d

        # Iterate through layers in reverse and backprop through them.
        for index, layer in enumerate(reversed(self.layers)):
            backInput = layer.backward(backInput)

        # Reproduce 3D volume.

        # First extract the batch size.
        backInput = backInput.reshape(backInput.shape[0], self.latestInput[0], -1).transpose(1, 0, 2)

        # Next extract the image using PyTorch's fold functionality.
        fold = torch.nn.Fold(output_size=self.latestInput[2:], kernel_size=self.f, padding=self.p, stride=self.s)
        result = fold(torch.from_numpy(backInput)).numpy()

        return result

    def update(self, lr, l2Reg, t=0):
        for layer in self.layers:
            layer.update(lr, l2Reg=l2Reg, t=t)

    def outputSize(self):
        return (self.depth2, self.height2, self.width2)