import numpy


#### Loss Functions ####

# Function that takes an output, implements the soft max function and cross entropy
# to calculate loss and error gradient. X is output, y is label.
# Significantly, y is a value, not a 1-hot encoded vector.
def softmaxLoss(X, y, eps= 1e-14):

    # Flip the input
    x = X.T

    # Softmax 'activation'
    probabilities = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    probabilities /= numpy.sum(probabilities, axis=1, keepdims=True)

    # Number of examples.
    N = x.shape[0]

    # Calculate loss by multidimensionally indexing the probabilities for each example and summing
    # the log of the correct output probability for all examples.
    loss = -numpy.sum(numpy.log(probabilities[range(N), y] + eps)) / N

    # Calculate gradient by deducting 1 from the correct class in the output X.
    dx = probabilities
    dx[range(N), y] -= 1
    dx /= N

    return loss, dx


#### Activation Functions ####

# RELU activation function
def relu(x):
    return numpy.maximum(0, x)

# Derivative of the RELU function for backpropagation.
def dRelu(x):
    return x > 0

# Sigmoid activation function.
def sigmoid(x):
    return x * (1 - x)

# Derivative of the Sigmoid activation function for backpropagation
def dSigmoid(x):
    return 1 / (1 + numpy.exp(-x))


#### Update Methods ####

# Vanilla regularised gradient descent
def vanillaUpdate(neurons, lr, l2Reg=0):

     for n in neurons:
         # Calculate regularisation of neuron in respect to weight
         l2 = l2Reg * n.weights
         # Calculate weight error gradient of neuron
         dx = (n.latestInput.dot(n.delta)).T + l2
         # Calculate bias error gradient of neuron
         dBias = numpy.sum(n.delta)

         # Update values
         n.weights -= lr * dx + l2
         n.bias -= lr * dBias

# Adam update method. Intersection fo RMSProp and Momentum updating. Values from beta1, beta2 amd epsilon
# shamelessly stolen from Adam paper.
def adamUpdate(neurons, lr, t, l2Reg=0, beta1=numpy.float32(0.9), beta2=numpy.float32(0.999), eps=1e-8):

    for n in neurons:

        # Same as vanilla update
        # Calculate regularisation of neuron in respect to weight
        l2 = l2Reg * n.weights

        # Calculate weight error gradient of neuron
        dx = (n.latestInput.dot(n.delta)).T + l2
        # Calculate bias error gradient of neuron
        dBias = numpy.sum(n.delta)

        # Stolen from here: https://gist.github.com/skaae/ae7225263ca8806868cb
        gamma = 1-eps
        beta1_t = beta1 * gamma ** (t-1)

        # Include momentum and RMSProp -> adaptive learning rates that change for each neuron
        # based on its gradients but not necessarily just reduced + lots and lots of maths.
        n.m = beta1_t * n.m + (1 - beta1) * dx
        n.v = beta2 * n.v + (1 - beta2) * dx ** 2

        # Momentum
        m = n.m / numpy.float32(1 - (beta1 ** t))
        # Velocity
        v = n.v / numpy.float32(1 - (beta2 ** t))

        # Now for update of values.
        n.weights -= lr * m / (numpy.sqrt(v) + eps)
        n.bias -= lr * dBias

# RMSProp method of neuron weight update. Decays update to reduce oscillations.
def RMSPropUpdate(neurons, lr, l2Reg=0, decayRate=0.9, eps=1e-8):

    for n in neurons:
        l2 = l2Reg * n.weights

        dx = (n.latestInput.dot(n.delta)).T + l2

        dBias = numpy.sum(n.delta)

        n.cache = decayRate * n.cache + (1 - decayRate) * (numpy.square(dx))

        n.weights += - lr * dx / (numpy.sqrt(n.cache) + eps)
        n.bias -= lr * dBias


#### Functions to flatten volumes to perform convolutions by matrix dot multiplication
#    rather than convoluted convolution operations.
####

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = numpy.repeat(numpy.arange(field_height), field_width)
    i0 = numpy.tile(i0, C)
    i1 = stride * numpy.repeat(numpy.arange(out_height), out_width)
    j0 = numpy.tile(numpy.arange(field_width), field_height * C)
    j1 = stride * numpy.tile(numpy.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = numpy.repeat(numpy.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):

    p = padding
    x_padded = numpy.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):

    N, C, H, W = x_shape

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = numpy.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    numpy.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

