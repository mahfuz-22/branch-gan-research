import os
import json
import pickle
import random
from google.colab import drive
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Configuration
MAX_SEQ_LENGTH = 128
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"
OUTPUT_DIR = "/content/drive/MyDrive/CustomPileData"
METADATA_FILE = "metadata.json"

# Dataset configuration - redistributed proportions among available datasets
DATASETS = {
    "wikipedia": {
        "name": "wikipedia",
        "config": "20220301.en",  # English Wikipedia
        "split": "train",
        "text_key": "text",
        "sample_size": 3000,  # ~25%
        "epochs": 2
    },
    "openwebtext": {
        "name": "openwebtext",
        "split": "train",
        "text_key": "text",
        "sample_size": 6000,  # ~50%
        "epochs": 1
    },
    "bookcorpus": {
        "name": "bookcorpus",
        "split": "train",
        "text_key": "text",
        "sample_size": 3000,  # ~25%
        "epochs": 2
    }
}

def mount_drive():
    """Mount Google Drive and create output directory"""
    drive.mount('/content/drive')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory at {OUTPUT_DIR}")

def get_data(dataset_config):
    """Load and sample data from a dataset"""
    try:
        if "config" in dataset_config:
            dataset = load_dataset(
                dataset_config["name"],
                dataset_config["config"],
                split=dataset_config["split"],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_config["name"],
                split=dataset_config["split"],
                trust_remote_code=True
            )

        # Select random samples
        total_size = len(dataset)
        indices = random.sample(range(total_size), min(dataset_config["sample_size"], total_size))
        samples = dataset.select(indices)

        texts = [sample[dataset_config["text_key"]] for sample in samples]

        # Repeat texts based on epochs
        texts = texts * dataset_config["epochs"]
        return texts

    except Exception as e:
        print(f"Error loading dataset {dataset_config['name']}: {str(e)}")
        return []

def tokenize_data(texts, tokenizer):
    """Tokenize texts with specified max length"""
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )["input_ids"]

def save_to_pickle(data, filename):
    """Save tokenized data to pickle file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving pickle file {filename}: {str(e)}")

def main():
    # Mount Google Drive
    mount_drive()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_data = []
    metadata = {}

    # Process each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        data = get_data(config)
        if data:
            print(f"Got {len(data)} samples from {dataset_name}")
            print("Sample text:")
            print(f"{data[0][:200]}...\n")
            all_data.extend(data)

    print(f"\nTotal samples gathered: {len(all_data)}")
    print("Shuffling data...")
    random.shuffle(all_data)

    print("Tokenizing data...")
    tokenized_data = tokenize_data(all_data, tokenizer)

    # Save in chunks
    chunk_size = 1000
    for i in tqdm(range(0, len(tokenized_data), chunk_size)):
        chunk = tokenized_data[i:i+chunk_size]
        filename = os.path.join(OUTPUT_DIR, f"chunk_{i//chunk_size}.pkl")
        save_to_pickle(chunk, filename)
        metadata[f"chunk_{i//chunk_size}.pkl"] = len(chunk)

    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    print(f"\nProcessed {len(all_data)} total samples")
    print(f"Created {len(metadata)} pickle files")
    print(f"Data saved to {OUTPUT_DIR}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()