import random
import numpy as np
from .neuron import Neuron

class Network:
    def __init__(self, layer_sizes):
        self.genes = []

        # We create a "blank" network (full of random numbers),
        # by making as many layers as there are items in layer_sizes (items in that list tell us size of layers).
        # Each layer consists of as many neurons as there are defined in layer sizes
        for i in range(1, len(layer_sizes)):
            # layer_sizes[i-1] = previous layer (input for current layer)
            # layer_sizes[i] = current layer
            
            layer = []
            for _ in range(layer_sizes[i]):
                # for now just                 list of random numbers (weights),                  a random numer (bias),  and activation function (max(0, x) for hidden layers, x for output layers)
                layer.append(Neuron(weights = [random.random() for _ in range(layer_sizes[i-1])], bias = random.random(), activation = self.get_activation(i, len(layer_sizes))))
            
            self.genes.append(layer)

    def predict(self, inputs:list):
        for layer in self.genes:
            # All neurons in a layer get the same inputs[which are values of neurons from previous layer], 
            # but because each neuron has different weights they calculate different number.
            # Their values are then stored in list which are then used for the nex layer
            outputs = [neuron.forward(inputs) for neuron in layer]
            inputs = outputs
        # Because the last layer consists of only 1 neuron this is just to return the price, not list with 1 price
        return inputs[0]
    
    def evaluate(self, dataset, uses_log_scaling = False):
        errors = []
        
        for input, target in dataset:
            prediction = self.predict(input)
            
            # If we use log-scaling we don't normalize the price (in the dataset)
            if uses_log_scaling:
                prediction = np.expm1(prediction)
                target = np.expm1(target)
                
            errors.append(abs(prediction - target))
        
        MAE = sum(errors) / len(errors) # Average error
        
        return MAE

    def get_genes(self):
        genes = []
        for layer in self.genes:
            layer_genes = []
            for neuron in layer:
                layer_genes.append([neuron.weights, neuron.bias])
            genes.append(layer_genes)
        return genes

    def set_genes(self, genes):
        for layer, layer_genes in zip(self.genes, genes):
            for neuron, (new_weights, new_bias) in zip(layer, layer_genes):
                neuron.weights = new_weights.copy()  # overwrite weights
                neuron.bias = new_bias               # overwrite bias

    def get_activation(self, index, total_layers):
        """Returns the correct activation function based on the rules we set."""
        if index == total_layers - 1:
            return lambda x:x # Linear for the output
        else:
            return lambda x: x if x > 0 else x * 0.1 # Leaky ReLU for the hidden layers
        
    def mutate_genes(self, genes, mutation_rate = 0.1, mutation_strength = 0.1, use_gaussian_dist = False):
        """
        Mutate a given set of genes (weights and biases) and return a new set of genes.
        
        Parameters:
        - genes: list of layers, each layer is a list of [weights, bias] pairs
        - mutation_rate: probability of mutating each weight/bias
        - mutation_strength: maximum mutation magnitude
        - use_gaussian_dist: if True, use Gaussian mutation; otherwise uniform
        
        Returns:
        - new_genes: mutated copy of the input genes
        """
        new_genes = []

        for layer in genes:
            layer_genes = []
            for weights, bias in layer:
                new_weights = []
                for w in weights:
                    if random.random() < mutation_rate:
                        if use_gaussian_dist:
                            w += random.gauss(0, mutation_strength)
                        else:
                            w += random.uniform(-mutation_strength, mutation_strength)
                    # Clamp weight
                    w = max(min(w, 5.0), -5.0)
                    new_weights.append(w)

                # Mutate bias
                if random.random() < mutation_rate:
                    if use_gaussian_dist:
                        bias += random.gauss(0, mutation_strength)
                    else:
                        bias += random.uniform(-mutation_strength, mutation_strength)
                # Clamp bias
                bias = max(min(bias, 5.0), -5.0)

                layer_genes.append([new_weights, bias])
            new_genes.append(layer_genes)

        return new_genes