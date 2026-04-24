"""

# steps to start training

you have a large dataset of nested dictionaries ready, the most critical next step is to ensure the vocabulary (stoi) and sequence length are perfectly tuned for your data. If your dataset is large, even small errors in character mapping can break the training.
Here are the three final steps to start training successfully:
1. Final Vocabulary Check
Before running the loader, make sure your stoi map covers every possible character in your dataset. If you have "hidden" characters or unique symbols, add them to your special variable.
"""

# Pro-tip: Create stoi dynamically from your entire dataset
full_text = "".join([str(d) for d in my_large_dataset]) 
chars = sorted(list(set(full_text)))
stoi = { ch:i for i,ch in enumerate(chars) }
print(f"Total Unique Characters: {len(chars)}")


"""
2. Handling Variable Nesting Depth
If some of your dictionaries are very deep and others are shallow, your sequences will vary greatly in length.
Check Max Length: Find the longest string in your flattened dataset and set max_len in the DictionaryDataset accordingly.
Memory Tip: If your sequences are longer than 1024 characters, the Transformer will use a lot of VRAM. You might need to reduce the batch_size in the DataLoader.
3. The Optimizer and Loss
For trend prediction or classification on this structured data:
"""

# If predicting categories (e.g., Trend Up, Trend Down, Stable)
criterion = torch.nn.CrossEntropyLoss()

# Recommended Optimizer for Transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

"""

4. GPU Acceleration
Ensure you move both the model and the data to the GPU (if available) for much faster training on a large dataset:
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for inputs, targets in loader:
    inputs, targets = inputs.to(device), targets.to(device)
    # ... rest of training ...
  
