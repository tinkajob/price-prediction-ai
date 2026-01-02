import pandas
import random

num_of_networks = 20

class Neuron:
    def __init__(self, weights:list, bias:float, activation:function):
        self.weights = weights
        self.bias = bias
        self.activation = activation #should be lambda x: max(0, x) for hidden layers and lambda x: x

    def forward(self, inputs:list):
        sum = 0
        for weight, input in zip(self.weights, inputs):
            sum += weight * input

        sum += self.bias
        return self.activation(sum)


class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        # We create a "blank" network (full of random numbers),
        # by making as many layers as there are items in layer_sizes (items in that list tell us size of layers).
        # Each layer consists of as many neurons as there are defined in layer sizes
        for i in range(1, len(layer_sizes)):
            # layer_sizes[i-1] = previous layer (input for current layer)
            # layer_sizes[i] = current layer
            
            layer = []
            for _ in range(layer_sizes[i]):
                # for now just                 list of random numbers (weights),                  a random numer (bias),  and activation function (max(0, x) for hidden layers, x for output layers)
                layer.append(Neuron(weights = [random.random() for _ in range(layer_sizes[i-1])], bias = random.random(), activation = lambda x: max(0, x) if i != len(layer_sizes) - 1 else lambda x: x))
            
            self.layers.append(layer)

    def predict(self, inputs:list):
        for layer in self.layers:
            # All neurons in a layer get the same inputs[which are values of neurons from previous layer], 
            # but because each neuron has different weights they calculate different number.
            # Their values are then stored in list which are then used for the nex layer
            outputs = [neuron.forward(inputs) for neuron in layer]
            inputs = outputs
        # Because the last layer consists of only 1 neuron this is just to return the price, not list with 1 price
        return inputs[0]
    
    def evaluate(self, dataset):
        errors = []
        for input, target in dataset:
            prediction = self.predict(input)
            errors.append(abs(prediction - target))
        MAE = sum(errors) / len(errors) # Average error
        return MAE


# We load the dataset, then we clean it
df = pandas.read_csv("houses.csv")
df = df.dropna()
df = df[df["price"] > 0]

training_data = df[:3200]
validation_data = df[3200:]

features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]
target = df["price"]

training_dataset = [(row[features].values.tolist(), row[target].values.tolist()) for _, row in training_data.iterrows()]
validation_dataset = [(row[features].values.tolist(), row[target].values.tolist()) for _, row in validation_data.iterrows()]

max_generations = 500

network_size = [len(features), 25, 1]
population = [Network(network_size) for _ in range(num_of_networks)]

# Training loop
for generation in range(max_generations):
    for network in population:
        network.evaluate(training_dataset)



    print(f"Completed training generation: {generation}")