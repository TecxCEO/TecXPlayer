# 1. Your raw data
my_dicts = [{"temp": 22, "status": {"fan": "on"}}, {"temp": 30, "status": {"fan": "off"}}]
my_labels = [0, 1] 

# 2. Initialize Dataset and Loader
dataset = DictionaryDataset(my_dicts, my_labels, stoi)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 3. Standard Training Loop
for inputs, targets in loader:
    outputs = model(inputs) # 'inputs' is now a clean batch of tensors
    # ... loss, backward, step ...

