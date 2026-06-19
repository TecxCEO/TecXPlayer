import json
import torch

# Load the JSON data
with open("training_data.json", "r") as f:
    data = json.load(f)

# Extract and flatten all tokens across 18 batches and 20 steps
all_tokens = []
for batch in data[0]["context_window_batches"]:
    for step in batch["steps"]:
        # Safety check: Ensure every step is exactly 44 tokens
        assert len(step["tokens"]) == 44, f"Error: Step layout must be 44 tokens! Got {len(step['tokens'])}"
        all_tokens.extend(step["tokens"])

# Convert to PyTorch Tensor for training
model_input_tensor = torch.tensor([all_tokens]) # Shape: (1, 15840)

print(f"Data verification successful!")
print(f"Total tokens in context window: {model_input_tensor.shape[1]}")

