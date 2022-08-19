import numpy as np

class Module:
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        self.updateInputGrad(input, gradOutput)
        self.updateAccGrad(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        return input

    def updateInputGrad(self, input, gradOutput):
        self.gradInput = gradOutput
        return gradOutput

    def updateAccGrad(self, input, gradOutput):
        pass

    def zeroGrad(self):
        pass

    def getParams(self):
        pass

    def getParamGrads(self):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __repr__(self):
        return "Module class"


class Sequential(Module):

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.outputs_ = []

    def updateOutput(self, input):
        for module in self.modules:
            self.outputs_.append(module.forward(input))
            input = self.outputs_[-1]

        return self.outputs_[-1]

    def backward(self, input, gradOutput):

        for idx in range(len(self.modules), 1, -1):
            grad = self.modules[idx-1].backward(self.outputs_[idx-2], gradOutput)
            gradOutput = grad

        grad = self.modules[0].backward(input, gradOutput)
        self.gradInput = grad

        return self.gradInput

    def getParams(self):
        return [x.getParams() for x in self.modules]

    def getParamGrads(self):
        return [x.getParamGrads() for x in self.modules]


    def zeroGrad(self):
        for module in self.modules:
            module.zeroGrad()

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()

    def __repr__(self):
        return "".join([str(x) for x in self.modules])

    def __getitem__(self, item):
        return self.modules.__getitem__(item)


class Linear(Module):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        stdv = 1/np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_in, n_out))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradB = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = self.W@input + self.b
        return self.output

    def updateInputGrad(self, input, gradOutput):
        self.gradInput = self.W@gradOutput

