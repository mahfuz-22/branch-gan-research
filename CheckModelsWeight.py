# Check Model's weight

def inspect_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(f"Mean: {param.data.mean().item()}")
            print(f"Std: {param.data.std().item()}")
            print(f"Min: {param.data.min().item()}")
            print(f"Max: {param.data.max().item()}")
            print("----------------------")

# Use the function
inspect_model_weights(trained_model.generator.generator)