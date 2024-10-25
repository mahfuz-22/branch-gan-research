# Testing some text generation with the trained model

# First mount Google Drive if not already mounted
from google.colab import drive
try:
    drive.mounted
except:
    drive.mount('/content/drive')
    drive.mounted = True

import os
import torch
import json
from datetime import datetime
from Models.GanSetup import GannSetup
from Models.Generator import NonResidualGenerator
from Models.Discriminator import NonResidualDiscriminatorWithDualValueHeads
from transformers import GPT2Config, AutoTokenizer

def list_available_models(base_dir="/content/drive/MyDrive/Branch-GAN-Models/experiments"):
    """List all available model versions with their timestamps."""
    if not os.path.exists(base_dir):
        print(f"No models found in {base_dir}")
        return []

    models = []
    for dirname in os.listdir(base_dir):
        if dirname.startswith("Branch-GAN-Pile-experiment_"):
            timestamp = dirname.split("_")[-2] + "_" + dirname.split("_")[-1]
            try:
                date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                models.append({
                    'dirname': dirname,
                    'timestamp': timestamp,
                    'datetime': date_obj,
                    'full_path': os.path.join(base_dir, dirname)
                })
            except ValueError:
                continue

    # Sort by datetime
    models.sort(key=lambda x: x['datetime'], reverse=True)

    print("\nAvailable models:")
    for idx, model in enumerate(models):
        print(f"{idx}. {model['dirname']} ({model['datetime'].strftime('%Y-%m-%d %H:%M:%S')})")

    return models

def select_model_version(models, selection=None):
    """Select a specific model version either by index or let user choose."""
    if not models:
        raise ValueError("No models available to select from")

    if selection is not None:
        if isinstance(selection, int):
            if 0 <= selection < len(models):
                return models[selection]['full_path']
            else:
                raise ValueError(f"Selection index {selection} out of range")
        elif isinstance(selection, str):
            # Try to find model by timestamp
            for model in models:
                if selection in model['dirname']:
                    return model['full_path']
            raise ValueError(f"No model found matching timestamp {selection}")

    # Interactive selection if no selection provided
    while True:
        try:
            idx = int(input("\nEnter the number of the model you want to load: "))
            if 0 <= idx < len(models):
                return models[idx]['full_path']
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_trained_model(model_path):
    """Load a trained model from the specified path."""
    # Load generator configuration
    generator_config_path = os.path.join(model_path, "Generator-Config.json")
    with open(generator_config_path, "r") as f:
        generator_config_dict = json.load(f)

    # Load discriminator configuration
    discriminator_config_path = os.path.join(model_path, "Discriminator-Config.json")
    with open(discriminator_config_path, "r") as f:
        discriminator_config_dict = json.load(f)

    # Create config objects
    generator_config = GPT2Config(**generator_config_dict)
    discriminator_config = GPT2Config(**discriminator_config_dict)

    # Create GannSetup with configs
    model = GannSetup(generator_config, discriminator_config)

    # Load generator weights
    generator_path = os.path.join(model_path, "Current", "Generator", "Generator.pt")
    generator_state_dict = torch.load(generator_path, map_location=torch.device('cpu'))
    model.generator.load_state_dict(generator_state_dict)

    # Load discriminator weights
    discriminator_path = os.path.join(model_path, "Current", "Discriminator", "Discriminator.pt")
    discriminator_state_dict = torch.load(discriminator_path, map_location=torch.device('cpu'))
    model.discriminator.load_state_dict(discriminator_state_dict)

    print("Model loaded successfully from:", model_path)
    return model

def generate_text(model, context, max_length=100, device='cpu'):
    """Generate text using the loaded model."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Move model to specified device
    model = model.to(device)

    # Encode the context
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    print(f"Input shape: {input_ids.shape}")

    # Generate
    with torch.no_grad():
        output = model.generator.generator.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,  # Enable sampling
            top_k=50,        # Consider top 50 tokens
            top_p=0.95       # Nucleus sampling
        )

    print(f"Output shape: {output.shape}")

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Move model back to CPU to free GPU memory
    model = model.to('cpu')

    return generated_text

# Example usage:
def test_model_generation(timestamp=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # List and select available models
    models = list_available_models()
    if timestamp:
        model_path = select_model_version(models, timestamp)
    else:
        model_path = select_model_version(models)

    # Load the selected model
    trained_model = load_trained_model(model_path)

    # Generate text
    context = "Fiction Context: then came the bats, and the star that was like ugh-lomi crept out of its blue hiding-place in the west. she called to"
    generated_text = generate_text(trained_model, context, device=device)

    print("\nGenerated text:")
    print(generated_text)