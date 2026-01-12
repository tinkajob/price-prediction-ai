import pandas
from utils.utils import *
from modules.normalizer import Normalizer
from modules.network import Network

parameters = load_json("train/training_parameters.json")

population_size = parameters["population_size"]
survivors_count = parameters["survivors_count"]

features = parameters["features"]
target = parameters["target"]

max_generations = parameters["max_generations"]

patience = parameters["patience"]

mutation_rate = parameters["mutation_rate"]
mutation_strength = parameters["mutation_strength"]

norm = Normalizer()

# We load the dataset, then we clean it
df = pandas.read_csv("houses.csv")
df = df.dropna()
df = df[df["price"] > 0]

training_data = df[:3200]
validation_data = df[3200:]

# We 'configure' normalizer only on training set, not on test set, and use only this configuration to normalize BOTH subsets (so that they are normalized in the same way)
norm.fit(df[:3200], features + target) 
training_data = norm.transform(training_data, features + target)
validation_data = norm.transform(validation_data, features + target)

# This are lists of tuples: ([a, b, c, d, e, f], g)
training_dataset = [(row[features].values.tolist(), row[target[0]]) for _, row in training_data.iterrows()]
validation_dataset = [(row[features].values.tolist(), row[target[0]]) for _, row in validation_data.iterrows()]

network_size = [len(features), 25, 1]
population = [Network(network_size) for _ in range(population_size)]

best_model_score = 99999999999
gens_without_improvement = 0
last_gen = 0

print("================================\n      STARTING TRAINING\n================================\n")

# Training loop
for generation in range(1, max_generations + 1):
    gen_performance = []
    for child in population:
        # This mae should be as low as possible
        norm_mae = child.evaluate(training_dataset)
        gen_performance.append((child, norm_mae))

    # After we have finished training a generation, we sort networks by their performance, and pick top n survivors
    gen_performance.sort(key = lambda x:x[1])
    last_gen = generation

    # To keep track of the best model so far
    if gen_performance[0][1] < best_model_score:
        best_model = gen_performance[0][0]
        best_model_score = gen_performance[0][1]
        gens_without_improvement = 0
    else:
        gens_without_improvement += 1
        if gens_without_improvement >= patience:
            print("MODEL EXCEEDED PATIENCE MAXIMUM!\nSTOPPING NOW")
            break

    best_mae = gen_performance[0][1]

    print(f"COMPLETED TRAINING GENERATION: {generation}")
    print(f"    - Best MAE (normalized): {best_mae}")
    print(f"    - Best MAE (dollars): {norm.invert_value(best_mae, "price"):,.2f}")
    print(f"    - Patience used: {gens_without_improvement}\n")

    survivors = [network for network, mae in gen_performance[:survivors_count]]

    # The non-survivors will be ovverwritten by copies of new mutations of survivors
    remaining = [network for network, mae in gen_performance[survivors_count:]]

    for child in remaining:
        parent = random.choice(survivors)
        genes = parent.get_genes()
        new_genes = mutate_genes(genes, mutation_rate = mutation_rate, mutation_strength = mutation_strength)
        child.set_genes(new_genes)

    population = survivors + remaining

best_model_genes = best_model.get_genes()
metrics = {"timestamp": datetime.now().isoformat(timespec="seconds").replace(":", "-"), "generation": last_gen}
save_model(best_model_genes, metrics, parameters, input("How would you like to name this model? (leave empty for default)\n: "))