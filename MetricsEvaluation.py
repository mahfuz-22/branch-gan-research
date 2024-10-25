# Get Automatic Metrics / Results

# First mount Google Drive if not already mounted
from google.colab import drive
try:
    drive.mounted
except:
    drive.mount('/content/drive')
    drive.mounted = True

import torch
from transformers import AutoTokenizer
import numpy as np
import os
from datetime import datetime
from Models.GanSetup import GannSetup
from Models.Generator import NonResidualGenerator
from Models.Discriminator import NonResidualDiscriminatorWithDualValueHeads
from transformers import GPT2Config

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

    while True:
        try:
            idx = int(input("\nEnter the number of the model you want to analyze: "))
            if 0 <= idx < len(models):
                return models[idx]['full_path']
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_trained_model(model_path, device='cpu'):
    """Load a trained model from the specified path."""
    # Load configurations
    generator_config_path = os.path.join(model_path, "Generator-Config.json")
    discriminator_config_path = os.path.join(model_path, "Discriminator-Config.json")

    with open(generator_config_path, "r") as f:
        generator_config_dict = json.load(f)
    with open(discriminator_config_path, "r") as f:
        discriminator_config_dict = json.load(f)

    # Create config objects and model
    generator_config = GPT2Config(**generator_config_dict)
    discriminator_config = GPT2Config(**discriminator_config_dict)
    model = GannSetup(generator_config, discriminator_config)

    # Load weights
    generator_path = os.path.join(model_path, "Current", "Generator", "Generator.pt")
    discriminator_path = os.path.join(model_path, "Current", "Discriminator", "Discriminator.pt")

    generator_state_dict = torch.load(generator_path, map_location=device)
    discriminator_state_dict = torch.load(discriminator_path, map_location=device)

    model.generator.load_state_dict(generator_state_dict)
    model.discriminator.load_state_dict(discriminator_state_dict)

    model = model.to(device)
    print("Model loaded successfully from:", model_path)
    return model

def calculate_metrics(model, tokenizer, context, max_length=100, num_samples=5, device='cpu'):
    """Calculate metrics for the model's text generation."""
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)

    metrics = {
        "perplexity": [],
        "token_entropy": [],
        "unique_tokens": [],
        "repetition_rate": []
    }

    generated_texts = []  # Store generated texts for later analysis

    for i in range(num_samples):
        try:
            with torch.no_grad():
                output = model.generator.generator.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                )

                generated_ids = output.sequences[0]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)

                # Calculate perplexity
                if hasattr(output, 'scores'):
                    scores = torch.stack(output.scores, dim=1)
                    log_likelihood = torch.nn.functional.log_softmax(scores, dim=-1)
                    sequence_log_likelihood = log_likelihood.gather(
                        dim=-1,
                        index=generated_ids[input_ids.shape[1]:].unsqueeze(-1)
                    ).squeeze(-1)
                    perplexity = torch.exp(-sequence_log_likelihood.mean())
                    metrics["perplexity"].append(perplexity.item())

                    # Calculate token entropy
                    probabilities = torch.softmax(scores, dim=-1)
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
                    metrics["token_entropy"].append(entropy.mean().item())
                else:
                    print("Warning: Scores not available. Skipping perplexity and entropy calculations.")

                # Calculate unique tokens
                unique_tokens = len(set(generated_ids.tolist()))
                metrics["unique_tokens"].append(unique_tokens)

                # Calculate repetition rate
                tokens = generated_ids.tolist()
                repetitions = sum(tokens[i] == tokens[i+1] for i in range(len(tokens)-1))
                repetition_rate = repetitions / (len(tokens) - 1)
                metrics["repetition_rate"].append(repetition_rate)

                print(f"\nGenerated text (sample {i+1}):\n{generated_text}\n")

        except Exception as e:
            print(f"Error in sample {i+1}: {str(e)}")

    # Average the metrics
    for key in metrics:
        if metrics[key]:  # Only calculate mean if there are values
            metrics[key] = np.mean(metrics[key])
            metrics[f"{key}_std"] = np.std(metrics[key]) if len(metrics[key]) > 1 else 0
        else:
            metrics[key] = None
            metrics[f"{key}_std"] = None

    return metrics, generated_texts

def analyze_model(timestamp=None, context=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Complete analysis pipeline for a selected model."""
    # List and select available models
    models = list_available_models()
    if timestamp:
        model_path = select_model_version(models, timestamp)
    else:
        model_path = select_model_version(models)

    # Load the selected model
    trained_model = load_trained_model(model_path, device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Use default context if none provided
    if context is None:
        context = "Fiction Context: then came the bats, and the star that was like ugh-lomi crept out of its blue hiding-place in the west. she called to"

    # Calculate metrics
    metrics, generated_texts = calculate_metrics(trained_model, tokenizer, context, device=device)

    # Print results
    print("\nMetrics:")
    for key, value in metrics.items():
        if "_std" not in key:  # Print mean and std together
            mean_value = value
            std_value = metrics.get(f"{key}_std")
            if mean_value is not None and std_value is not None:
                print(f"{key}: {mean_value:.4f} ± {std_value:.4f}")
            else:
                print(f"{key}: {mean_value}")

    return metrics, generated_texts, trained_model

# Example usage:
# analyze_model()  # Interactive selection
# analyze_model(timestamp="20240122_143022")  # Specific model version