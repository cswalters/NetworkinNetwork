import numpy

from neuralLayer import NeuralLayer
from convLayer import ConvLayer
from poolLayer import PoolLayer
from mlpLayer import MLPLayer
import util


class NeuralNetwork:

    def __init__(self, inputShape, layers, lr, l2Reg=0, loss='softmax'):

        self.layers = []
        self.lr = lr
        self.l2Reg = l2Reg
        self.lossMethod = loss
        self.epochCount = 0

        # If using dropout. Will get to this.
        self.dropoutMasks = []

        # t for Adam update method. Counts number of iterations.
        self.t = 0

        # Initial layer's input.
        nextInputSize = inputShape

        for layer in layers:
            # Generate the appropriate layers for network.
            if layer['type'] == 'neural':
                layer.pop('type')
                neural = NeuralLayer(nextInputSize, **layer)
                self.layers.append(neural)
                nextInputSize = neural.outputSize()

            elif layer['type'] == 'conv':
                layer.pop('type')
                conv = ConvLayer(nextInputSize, **layer)
                self.layers.append(conv)
                nextInputSize = conv.outputSize()

            elif layer['type'] == 'pool':
                layer.pop('type')
                pool = PoolLayer(nextInputSize, **layer)
                self.layers.append(pool)
                nextInputSize = pool.outputSize()

            elif layer['type'] == 'mlp':
                layer.pop('type')
                mlp = MLPLayer(nextInputSize, **layer)
                self.layers.append(mlp)
                nextInputSize = mlp.outputSize()

            elif layer['type'] == 'output':
                layer.pop('type')

                output = NeuralLayer(nextInputSize, **layer)
                output.isOutputLayer = True
                # Will use softmax on result instead
                output.useActivation = False

                self.layers.append(output)
                nextInputSize = output.outputSize()

    def predict(self, batch, label):

        nextInput = batch

        for index, layer in enumerate(self.layers):
            nextInput = layer.predict(nextInput)

        result = numpy.array(nextInput)

        if self.lossMethod == 'softmax':
            loss, delta = util.softmaxLoss(result, label)
        else:
            raise ValueError("Unspecified or invalid loss function")

        maxResult = numpy.argmax(result, axis=0)
        # Determine how many of the results matched the label.
        correctResult = numpy.sum(maxResult == label)

        return loss, correctResult / float(len(maxResult)) * 100

    def epoch(self, batch, label):

        #### Forward propagation ####

        l2 = 0
        nextInput = batch

        for index, layer in enumerate(self.layers):

            # Get tuple of result and l2 value
            layerResult = layer.forward(nextInput)

            # Store result for next layer to process
            nextInput = layerResult[0]

            # Update regularisation
            l2 += layerResult[1]

            # If using dropout, apply dropout mask to result before passing to next layer.
            # Note here inverted dropout is being used. The dropout mask is being divided by the p value (layer.dropout)
            if layer.dropout < 1 and not layer.isOutputLayer:
                dropoutMask = (numpy.random.rand(*nextInput.shape) < layer.dropout) / layer.dropout
                nextInput *= dropoutMask

                self.dropoutMasks.append(dropoutMask)

        result = numpy.array(nextInput)

        # Apply softmax loss to result.
        if self.lossMethod == 'softmax':
            loss, delta = util.softmaxLoss(result, label)
        else:
            raise ValueError("Unspecified or invalid loss function")

        # Adjust loss by l2 regularisation value.
        loss += 0.5 * self.l2Reg * l2

        maxResult = numpy.argmax(result, axis=0)
        # Determine how many of the results matched the label.
        correctResult = numpy.sum(maxResult == label)


        #### Backpropagation ####

        # Transpose output.
        backInput = delta.T

        # Iterate through layers in reverse and backprop through them.
        for index, layer in enumerate(reversed(self.layers)):

            # Check if no further back prop needed.
            notInputLayer = index < len(self.layers) - 1

            # Dropout errors that due to output being dropped-out in forward prop.
            if layer.dropout < 1 and not layer.isOutputLayer and self.dropoutMasks:

                dropoutMask = self.dropoutMasks.pop()

                # in the case of a flattened result being backprop'ed, adjust mask to match dimensions.
                if dropoutMask.ndim > 2 and backInput.ndim == 2:
                    # Transpose dropoutMask as we transpose backInput.
                    backInput *= dropoutMask.T.reshape(-1, backInput.shape[1])
                else:
                    backInput *= dropoutMask

            backInput = layer.backward(backInput, needNextDelta=notInputLayer)


        #### Update ####
        for index, layer in enumerate(self.layers):
            layer.update(self.lr, l2Reg=self.l2Reg, t=self.t)

        return loss + self.l2Reg * l2/2, correctResult / float(len(maxResult)) * 100




