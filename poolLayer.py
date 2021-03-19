import torch

from layer import Layer


class PoolLayer(Layer):

    def __init__(self, inputSize, f=2, s=2, method='max', dropout=1):

        # Captures the number of examples in a batch, initially 0
        self.imageSize = 0

        # Store layer's input dimensions
        self.depth = inputSize[0]
        self.height = inputSize[1]
        self.width = inputSize[2]

        self.method = method
        self.dropout = dropout

        # Store filter size and stride
        self.f = f
        self.s = s

        self.indices = None

        # Initialise PyTorch pooling and unpooling objects.
        if self.method == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=self.f, stride=self.s, return_indices=True)
            self.unpool = torch.nn.MaxUnpool2d(kernel_size=self.f, stride=self.s)
        else:
            raise ValueError("Unspecified or incorrect PoolLayer type.")

        # Ensure that the height and width are compatible with the pool filter size.
        assert self.height % self.f == 0
        assert self.width % self.f == 0

        self.height2 = (self.height - self.f) // self.s + 1
        self.width2 = (self.width - self.f) // self.s + 1

    def predict(self, batch):
        self.imageSize = batch.shape[0]
        return self.forward(batch)[0]

    def forward(self, batch):

        # Store number of examples in batch that are being processed.
        self.imageSize = batch.shape[0]

        # Pool volume and store indices of max values.
        result, self.indices = self.pool(torch.from_numpy(batch))

        # Return result
        return result.numpy(), 0

    def backward(self, d, needNextDelta=True):

        # If input has been flattened, revolumise it.
        if d.ndim < 4:
            d = d.reshape(self.imageSize, self.depth, self.height2, self.width2)

        # Unpool object using stored indices.
        delta = self.unpool(torch.from_numpy(d), self.indices).numpy()

        return delta

    def outputSize(self):
        return (self.depth, self.height2, self.width2)

    def update(self, lr, l2Reg, t=0):
        pass
