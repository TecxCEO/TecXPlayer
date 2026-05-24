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
  
"""
To find the perfect max_len, you need to know how long your dictionaries become once they are flattened into strings. Setting max_len to the 95th percentile is usually best; it captures almost all data without wasting memory on a few extremely long outliers.
1. Length Analysis Script
Run this on your large dataset to get the statistics:
"""

import numpy as np

def calculate_stats(dataset):
    def flatten(data):
        if isinstance(data, dict):
            return " ".join([f"{k}:{flatten(v)}" for k, v in data.items()])
        elif isinstance(data, list):
            return ",".join([flatten(i) for i in data])
        return str(data)

    # Calculate lengths of all flattened strings
    lengths = [len(flatten(d)) for d in dataset]

    avg_len = np.mean(lengths)
    max_len = np.max(lengths)
    p95 = np.percentile(lengths, 95)

    print(f"Average Length: {avg_len:.2f}")
    print(f"95th Percentile (Recommended max_len): {p95:.2f}")
    print(f"Absolute Max Length: {max_len}")
    
    return int(p95)

# Usage
# my_max_len = calculate_stats(my_large_dataset)

"""

2. Why the 95th Percentile?
Memory Efficiency: Transformers use 



 memory relative to sequence length. Reducing max_len from 2000 to 500 can make your training 16x more memory efficient.
Training Speed: Shorter sequences allow for larger Batch Sizes, which speeds up the GPU's work.
3. Procedural Steps for Setting max_len
Flatten Sample: Convert your dictionaries using the flatten logic provided.
Calculate Distributions: Use NumPy to find where most of your data lengths fall.
Set Threshold: Assign the p95 value to the max_len parameter in your DictionaryDataset.
Monitor Truncation: If the model performs poorly, check if critical information is being cut off at the end of long dictionaries.


#####


Adding special tokens is a pro move. It tells the model exactly where a dictionary starts (<SOS>) and ends (<EOS>), which is vital for nested structures.
1. Update your Vocabulary
You need to add three new tokens to your stoi map:
0: Padding (<PAD>) - for alignment.
1: Start of Sequence (<SOS>) - "Attention starts here."
2: End of Sequence (<EOS>) - "Pattern ends here.
""""


# Create new stoi with special tokens
chars = sorted(list(set(full_text)))
stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
stoi['<PAD>'] = 0
stoi['<SOS>'] = 1
stoi['<EOS>'] = 2

itos = {i: ch for ch, i in stoi.items()} # Reverse map


"""
2. Update the __getitem__ logic
Now, update your DictionaryDataset to "sandwich" the encoded characters between the SOS and EOS tokens:
"""

def __getitem__(self, idx):
    raw_string = self.flatten(self.data_list[idx])
    
    # 1. Start with SOS token
    encoded = [self.stoi['<SOS>']] 
    
    # 2. Add characters (handle unknown chars with <PAD> or 0)
    encoded += [self.stoi.get(c, 0) for c in raw_string]
    
    # 3. Add EOS token
    encoded.append(self.stoi['<EOS>'])
    
    # 4. Truncate if too long (keep EOS at the end)
    if len(encoded) > self.max_len:
        encoded = encoded[:self.max_len-1] + [self.stoi['<EOS>']]
        
    return torch.tensor(encoded), torch.tensor(self.labels[idx])


"""
3. Update the Model
Since you added 3 new tokens, ensure your vocab_size reflects this:

"""
# Use the length of your new stoi
model = DictionaryTransformer(vocab_size=len(stoi), ...)

"""
Why this helps:
Context: The model learns that anything following <SOS> is the start of a new data object.
Sequence Length: In a large dataset, this prevents the model from getting "lost" when sequences are very long or deeply nested.

####

To track if your model is actually learning or just memorizing (overfitting), you need a validation loop. This runs after every "epoch" (full pass through the data) using a separate set of data the model hasn't seen before.
1. Split Your Data
Before starting, split your large dataset into Training and Validation sets (usually an 80/20 split).
"""

from sklearn.model_selection import train_test_split

# Split your dictionaries and labels
train_data, val_data, train_labels, val_labels = train_test_split(
    my_large_dataset, my_labels, test_size=0.2, random_state=42
)

# Create loaders
train_loader = DataLoader(DictionaryDataset(train_data, train_labels, stoi), batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(DictionaryDataset(val_data, val_labels, stoi), batch_size=32, shuffle=False, collate_fn=collate_fn)



"""

2. The Training and Validation Loop
This script runs the training, then switches the model to eval() mode to test it.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- VALIDATION PHASE ---
    model.eval() # Disables dropout/batchnorm for testing
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad(): # No gradient calculation (saves memory)
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")


"""
3. What to watch for:
Overfitting: If Train Loss goes down but Val Loss starts going up, your model is memorizing the training data. You may need to add "Dropout" or more data.
Accuracy: If your accuracy stays at 50% (for 2 classes), the model isn't learning. Check your flatten logic or stoi mapping.
"""
