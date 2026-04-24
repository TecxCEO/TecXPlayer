"""

To export a Transformer model to ONNX, we have to handle the fact that Transformers often use variable-length sequences. We will use dynamic axes so your model can accept dictionaries of different sizes (e.g., a small dictionary vs. a large nested one) after export.
1. The Export Script
Run this script to convert your PyTorch .pth file into a universal .onnx file.
"""
import torch

def export_to_onnx(model, stoi, save_path="dict_model.onnx", max_len=512):
    model.eval()
    
    # 1. Create a dummy input (Batch size 1, Sequence length 'max_len')
    # We use integers because the model expects token indices
    dummy_input = torch.randint(0, len(stoi), (1, max_len)).to(next(model.parameters()).device)

    # 2. Export the model
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        export_params=True,        # Store the trained parameter weights inside the file
        opset_version=14,          # Use a modern opset for Transformer support
        do_constant_folding=True,  # Optimize the graph
        input_names=['input'],     # Name the input node
        output_names=['output'],   # Name the output node
        dynamic_axes={             # Allow variable batch sizes and sequence lengths
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {save_path}")

# Usage
# export_to_onnx(model, stoi)

"""

2. Why use ONNX for your Dictionary Model?
Speed: ONNX Runtime can be 2x to 5x faster than PyTorch for inference, especially on CPUs.
Portability: You can now run your dictionary-trend model in C++, Java (Android), or JavaScript (Browser) without needing to install PyTorch.
Optimization: The do_constant_folding flag removes unnecessary calculations in your nested Transformer layers, making the file smaller.
3. Verification
Once exported, you can verify the model using the onnx library:
"""

import onnx
onnx_model = onnx.load("dict_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

"""
4. Important Limitation
ONNX only exports the Neural Network. It does not export your Python flatten function or your stoi dictionary. When you use the model in a new app, you must still:
Flatten the dictionary to a string.
Convert characters to numbers using the exact same stoi map
"""
