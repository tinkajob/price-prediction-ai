import json, os
from datetime import datetime

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
