class Neuron:
    def __init__(self, weights:list, bias:float, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation #should be lambda x: max(0, x) for hidden layers and lambda x: x

    def forward(self, inputs:list):
        sum = 0
        for weight, input in zip(self.weights, inputs):
            sum += weight * input

        sum += self.bias
        return self.activation(sum)