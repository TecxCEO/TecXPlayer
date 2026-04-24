"""
To quantize Transformer model, use Post-Training Dynamic Quantization. This is the recommended method for Transformer-based models because it handles varying activation ranges effectively without needing a separate calibration dataset. 

ONNX Runtime

1. Quantization Script
This script uses onnxruntime.quantization to convert your weights from 32-bit floats (FP32) to 8-bit integers (INT8), typically reducing the file size by ~75%.
"""

# Post-Training Dynamic Quantization.

from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_path, output_path):
    # Perform dynamic quantization
    quantize_dynamic(
        model_input=input_path,      # Your original .onnx file
        model_output=output_path,    # New filename for the smaller model
        weight_type=QuantType.QInt8  # Quantize weights to INT8
    )
    print(f"Quantized model saved to: {output_path}")

# Usage
# quantize_onnx_model("dict_model.onnx", "dict_model_quantized.onnx")


"""
2. Benefits for Deployment
File Size: A 100MB model will drop to about 25MB, making it much easier to bundle into a mobile app.
Inference Speed: On most CPUs, INT8 operations are significantly faster than FP32, often resulting in 20-40% faster predictions.
Accuracy: For Transformer models, dynamic quantization typically sees only a minimal drop in accuracy (usually < 1%). 
YouTube
YouTube
3. Key Differences for Transformers
Feature 	Dynamic (Recommended)	Static
Best For	Transformers & RNNs	CNNs (Vision models)
Calibration	None required	Needs 100-1000 samples
Accuracy	Adapts to each input	Fixed parameters
Overhead	Slight runtime calculation	Purely optimized speed
4. Running the Quantized Model
Once quantized, you can load and run the .quantized.onnx file using the exact same code as your standard ONNX inference. ONNX Runtime automatically detects the INT8 format and uses the faster hardware instructions. 
 """
