import pandas
import random
import json

population_size = 20
survivors_count = 8

def get_activation(index, total_layers):
    """Returns the correct activation function based on the rules we set."""
    if index == total_layers - 1:
        return lambda x:x # Linear for the output
    else:
        return lambda x: max(0,x) # ReLU for the hidden layers

def mutate_genes(genes, mutation_rate = 0.1, mutation_strength = 0.1):
    new_genes = []
    for layer in genes:
        layer_genes = []
        for neuron in layer:
            new_weights = []
            new_bias = neuron[1]
            for weight in neuron[0]:
                if random.random() < mutation_rate:
                    weight += random.gauss(0, mutation_strength)
                new_weights.append(weight)
            if random.random() < mutation_rate:
                new_bias += random.gauss(0, mutation_strength)
            layer_genes.append((new_weights, new_bias))
        new_genes.append(layer_genes)
    return new_genes

def save_genes_to_json(genes):
    with open("models/best_network.json", "w") as f:
        json.dump(genes, f)

class Normalizer:
    def __init__(self):
        self.means = {} # (key, value) pairs, consisting of a feature and it's average value across the dataset 
        self.stds = {} # (key, value) pairs, consisting of a feature and it's standard deviation (how close the values are to the average)

    def fit(self, df, features):
        """Sets the values for transforming data"""
        # Here we find average value and standard deviation for each feature
        for feature in features:
            self.means[feature] = df[feature].mean()
            self.stds[feature] = df[feature].std() if df[feature].std() != 0 else 1 # To avoid division by 0

    def transform(self, df, features):
        """Calculates the normalized values"""
        df = df.copy() # To prevent modifying the existing dataframe
        for feature in features:
            df[feature] = (df[feature] - self.means[feature]) / self.stds[feature]
        return df
    
    def invert_transform(self, df, features):
        """Inverts the normalized values back to 'original'."""
        df = df.copy()
        for feature in features:
            df[feature] *= self.stds[feature]
            df[feature] += self.means[feature]
        return df
    
    def invert_value(self, value, feature):
        """Inverts the normalized value back to 'original' scale."""
        return value * self.stds[feature] + self.means[feature]

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
                layer.append(Neuron(weights = [random.random() for _ in range(layer_sizes[i-1])], bias = random.random(), activation = get_activation(i, len(layer_sizes))))
            
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
    
    def evaluate(self, dataset):
        errors = []
        
        for input, target in dataset:
            prediction = self.predict(input)
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

norm = Normalizer()

# We load the dataset, then we clean it
df = pandas.read_csv("houses.csv")
df = df.dropna()
df = df[df["price"] > 0]

features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]
target = ["price"]

training_data = df[:3200]
validation_data = df[3200:]

# We 'configure' normalizer only on training set, not on test set, and use only this configuration to normalize BOTH subsets (so that they are normalized in the same way)
norm.fit(df[:3200], features + target) 
training_data = norm.transform(training_data, features + target)
validation_data = norm.transform(validation_data, features + target)

# This are lists of tuples: ([a, b, c, d, e, f], g)
training_dataset = [(row[features].values.tolist(), row[target[0]]) for _, row in training_data.iterrows()]
validation_dataset = [(row[features].values.tolist(), row[target[0]]) for _, row in validation_data.iterrows()]

max_generations = 500

network_size = [len(features), 25, 1]
population = [Network(network_size) for _ in range(population_size)]

best_model_score = 99999999999

# Training loop
for generation in range(max_generations):
    gen_performance = []
    for child in population:
        # This mae should be as low as possible
        norm_mae = child.evaluate(training_dataset)
        gen_performance.append((child, norm_mae))

    print(f"COMPLETED TRAINING GENERATION: {generation}")

    # After we have finished training a generation, we sort networks by their performance, and pick top n survivors
    gen_performance.sort(key = lambda x:x[1])

    # To keep track of the best model so far
    if gen_performance[0][1] < best_model_score:
        best_model = gen_performance[0][0]
        best_model_score = gen_performance[0][1]

    best_mae = gen_performance[0][1]

    print(f"    - Best MAE (normalized): {best_mae}")
    print(f"    - Best MAE (dollars): {norm.invert_value(best_mae, "price")}")
    print()

    survivors = [network for network, mae in gen_performance[:survivors_count]]


    # The non-survivors will be ovverwritten by copies of new mutations of survivors
    remaining = [network for network, mae in gen_performance[survivors_count:]]

    for child in remaining:
        parent = random.choice(survivors)
        genes = parent.get_genes()
        new_genes = mutate_genes(genes, mutation_rate = 0.1, mutation_strength = 0.1)
        child.set_genes(new_genes)

    population = survivors + remaining

best_model_genes = best_model.get_genes() # Now we only need to save that thing
save_genes_to_json(best_model_genes)