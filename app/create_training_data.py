import json

def create_training_data():
    dataset_path = "app/data/text_classification_dataset.json"
    with open(dataset_path, "r") as file:
        data = json.load(file)
    return [(entry["text"], entry["label"]) for entry in data]
