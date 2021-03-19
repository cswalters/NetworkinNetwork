from abc import ABCMeta, abstractmethod


class Layer:

    __metaclass__ = ABCMeta

    isOutputLayer = False
    useActivation = True
    updateMethod = 'adam'  # Or can use vanilla SGD
    useDropout = 1

    @abstractmethod
    def predict(self, batch):
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def backward(self, d, needNextDelta=True):
        pass

    @abstractmethod
    def outputSize(self):
        pass

    @abstractmethod
    def update(self, lr, l2Reg, t=0):
        pass
