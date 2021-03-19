import numpy


class Neuron:

    # Initialises a naive neuron.
    def __init__(self, inputSize, isConv=False, bias=0.0):

        # Generate the weights with a standard deviation of 2.0 / number of inputs.
        self.weights = (numpy.random.randn(inputSize) * numpy.sqrt(2.0/inputSize)).astype(numpy.float32)
        self.bias = numpy.float32(bias)
        self.latestInput = None
        self.delta = None
        self.isConv = isConv

    # Determines, for an input, the activation of this neuron; the strength of its response.
    def strength(self, values):
        return numpy.dot(self.weights, values) + self.bias

    # Squares all weights and then sums them to produce regularisation value
    def regularisation(self):
        return numpy.sum(numpy.square(self.weights))