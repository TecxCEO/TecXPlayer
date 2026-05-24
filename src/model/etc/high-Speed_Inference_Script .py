"""

To run your exported Transformer model with high performance, use the onnxruntime library. This allows you to perform inference without needing PyTorch installed, which is ideal for deployment. 
ONNX Runtime
ONNX Runtime
 +3
1. Install ONNX Runtime 
Choose the version based on your hardware: 
ONNX Runtime
ONNX Runtime
 +1
For CPU: pip install onnxruntime
For GPU: pip install onnxruntime-gpu 
2. High-Speed Inference Script 
This script handles the Flattening and Encoding before feeding data into the optimized ONNX engine. 
"""

import onnxruntime as ort
import numpy as np

# 1. Load the ONNX session
# Choose 'CPUExecutionProvider' or 'CUDAExecutionProvider' for GPU
session = ort.InferenceSession("dict_model.onnx", providers=['CPUExecutionProvider'])

def run_live_inference(nested_dict, stoi, max_len=512):
    # --- REUSE YOUR FLATTEN LOGIC ---
    def flatten(data):
        if isinstance(data, dict):
            return " ".join([f"{k}:{flatten(v)}" for k, v in data.items()])
        elif isinstance(data, list):
            return ",".join([flatten(i) for i in data])
        return str(data)

    # --- PRE-PROCESS ---
    raw_str = flatten(nested_dict)
    encoded = [stoi['<SOS>']] + [stoi.get(c, 0) for c in raw_str] + [stoi['<EOS>']]
    
    # Pad to max_len (ONNX expects fixed or dynamic shapes, but consistent arrays are safer)
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len-1] + [stoi['<EOS>']]

    # --- RUN ONNX ---
    # ONNX expects NumPy arrays instead of PyTorch Tensors
    input_data = np.array([encoded], dtype=np.int64) 
    
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs) # None returns all outputs
    
    # Get the winning class index
    predicted_index = np.argmax(outputs[0], axis=1)[0]
    return predicted_index

# Example:
# result = run_live_inference(my_dict, stoi)

"""

Why use this for Dictionary Data?
Efficiency: ONNX Runtime optimizations like constant folding and node fusions specifically speed up Transformer "Attention" layers.
Memory: It often uses significantly less RAM than PyTorch (e.g., 80MB vs 370MB for similar models).
Dynamic Inputs: Because we exported with dynamic_axes, this session can handle dictionaries of varying lengths without crashing. 

Performance Monitoring
If your dictionary data is extremely large, you can further accelerate it by:
Quantization: Converting the model to INT8 to run faster on mobile or server CPUs.
IO Binding: If using a GPU, use IOBinding to keep the data on the device between steps.
"""
