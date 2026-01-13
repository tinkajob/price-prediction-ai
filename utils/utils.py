import random,json, os
from datetime import datetime

def mutate_genes(genes, mutation_rate = 0.1, mutation_strength = 0.1):
    new_genes = []
    for layer in genes:
        layer_genes = []
        for neuron in layer:
            new_weights = []
            new_bias = neuron[1]
            for weight in neuron[0]:
                if random.random() < mutation_rate:
                    weight += random.uniform(-mutation_strength, mutation_strength)
                new_weights.append(weight)
            if random.random() < mutation_rate:
                new_bias += random.uniform(-mutation_strength, mutation_strength)
            layer_genes.append((new_weights, new_bias))
        new_genes.append(layer_genes)
    return new_genes

def load_json(path):
    with open(path) as file:
        return json.load(file)

def save_model(genes, metrics, parameters, name = "", base_dir = "models"):
    """Saves model info in it's folder, along with the metadata. If the name of the model directory is name if it is given, otherwise timestamp is used."""

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    model_dir = os.path.join(base_dir, name if name != "" else f"model_{timestamp}")

    os.makedirs(model_dir, exist_ok = False)

    with open(os.path.join(model_dir, "genes.json"), "w") as f:
        json.dump(genes, f)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    with open(os.path.join(model_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f)
