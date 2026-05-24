"""
This function is the bridge between your raw data and the trained AI. It takes a Python dictionary, applies the same flattening, encoding, and special tokens used during training, and returns the model's prediction.
The Inference Function
"""


def predict_nested_dict(model, nested_dict, stoi, device, max_len=512):
    model.eval() # 1. Set to evaluation mode
    
    # 2. Flatten the dictionary using your established logic
    def flatten(data):
        if isinstance(data, dict):
            return " ".join([f"{k}:{flatten(v)}" for k, v in data.items()])
        elif isinstance(data, list):
            return ",".join([flatten(i) for i in data])
        return str(data)
    
    raw_string = flatten(nested_dict)
    
    # 3. Encode with SOS, EOS, and handle unknown characters
    encoded = [stoi['<SOS>']]
    encoded += [stoi.get(c, stoi.get('<PAD>', 0)) for c in raw_string]
    encoded.append(stoi['<EOS>'])
    
    # 4. Truncate/Pad
    if len(encoded) > max_len:
        encoded = encoded[:max_len-1] + [stoi['<EOS>']]
    
    # 5. Convert to Tensor and add Batch Dimension (1, seq_len)
    input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # 6. Run through model
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    
    return predicted_class.item()

# Example Usage:
# live_data = {"sensor": "A1", "readings": [10.5, 12.2], "alert": False}
# result = predict_nested_dict(model, live_data, stoi, device)
# print(f"Predicted Category: {result}")


"""
Tips for Best Results
Consistency: Ensure the flatten logic here is identical to the one used in your DataLoader. If you change a colon to a dash, the model will get confused.
Post-Processing: If your model predicts numbers (0, 1, 2), you can map them back to words using a simple list: categories = ["Low", "Medium", "High"].
Batching: If you need to predict thousands of dictionaries at once, it's faster to put them in a DataLoader rather than calling this function in a loop.
"""
