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
