"""
ONNX Export Utilities for AI-Blueprints

Simple API for converting various model formats to ONNX format.
Supports TensorFlow/Keras, NeMo, PyTorch, and Hugging Face Transformers models.
Automatically uses external weights files for models larger than 2GB.

Usage:
    from ais_utils.onnx_export import (
        export_tensorflow_to_onnx, 
        export_nemo_to_onnx,
        export_pytorch_to_onnx,
        export_transformers_to_onnx,
        export_sklearn_to_onnx
    )
    
    # Convert a trained TensorFlow/Keras model
    export_tensorflow_to_onnx("model.keras", input_shape=(1, 28, 28, 1), output_path="model.onnx")
    
    # Convert a NeMo model
    export_nemo_to_onnx("asr_model.nemo", input_sample={}, output_path="asr_model.onnx")
    
    # Convert a PyTorch model
    export_pytorch_to_onnx("model.pt", input_sample=torch.randn(1, 3, 224, 224), output_path="model.onnx")
    
    # Convert a Hugging Face model
    export_transformers_to_onnx("bert-base-uncased", task="text-classification", output_path="bert.onnx")
    
    # Convert a Scikit-learn model
    export_sklearn_to_onnx("model.pkl", input_sample=[[1, 2, 3, 4]], output_path="sklearn_model.onnx")
"""

import os
import warnings
from typing import Any, Optional, Tuple, Union, List, Dict
from pathlib import Path
import tempfile

# Optional imports - will be imported when needed
try:
    import numpy as np
    # Use np.ndarray in type hints only if numpy is available
    NDArray = np.ndarray
except ImportError:
    np = None
    # Fallback type for when numpy is not available
    NDArray = Any


def _save_with_external_data(onnx_model: Any, output_path: str, size_threshold: int = 2 * 1024 * 1024 * 1024) -> None:
    """
    Save ONNX model with external data for large models (>2GB).
    
    Args:
        onnx_model: ONNX model to save
        output_path: Path to save the model
        size_threshold: Size threshold in bytes (default 2GB)
    """
    try:
        import onnx
        
        # Get model size
        model_size = len(onnx_model.SerializeToString())
        
        if model_size > size_threshold:
            # Use external data for large models
            onnx.save_model(
                onnx_model, 
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_path).stem}.onnx.data",
                size_threshold=1024,  # Save tensors > 1KB externally
                convert_attribute=False
            )
            print(f"Large model saved with external weights: {output_path}")
        else:
            # Save normally for smaller models
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved to ONNX: {output_path}")
            
    except ImportError:
        # Fallback to basic save if onnx package not available
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Model saved to ONNX (fallback): {output_path}")


def export_tensorflow_to_onnx(model_path: str, input_shape: Optional[Tuple] = None,
                              output_path: str = "tensorflow_model.onnx",
                              model_name: str = "tensorflow_model") -> str:
    """
    Export a TensorFlow/Keras model to ONNX format with external weights support.
    
    Args:
        model_path: Path to saved Keras/TensorFlow model file (.keras, .h5, SavedModel dir)
        input_shape: Input shape tuple (batch_size, ...). If None, inferred from model
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to saved ONNX model
    """
    try:
        import tf2onnx
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        import tempfile
        
        # Load model from file path
        model = load_model(model_path)
        
        # Infer input shape if not provided
        if input_shape is None:
            input_shape = model.input_shape
            
        print(f"Converting model with input shape: {input_shape}")
        
        try:
            # Method 1: Direct conversion with patched model
            print("Attempting direct conversion from Keras model...")
            
            # Patch the model to add output_names if missing (Keras 3 compatibility)
            if not hasattr(model, 'output_names'):
                if hasattr(model, 'output'):
                    if isinstance(model.output, list):
                        model.output_names = [f"output_{i}" for i in range(len(model.output))]
                    else:
                        model.output_names = ['output']
                else:
                    model.output_names = ['output']
            
            # Create input signature
            spec = tf.TensorSpec(input_shape, tf.float32, name="input")
            
            # Convert using tf2onnx with input signature
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, 
                input_signature=[spec], 
                opset=12
            )
            print("✅ Direct conversion successful!")
            
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            print("Trying SavedModel approach...")
            
            # Method 2: SavedModel fallback with explicit signatures
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, "temp_saved_model")
                
                # Create a wrapper function for Sequential models
                @tf.function
                def model_func(x):
                    return model(x)
                
                # Get concrete function with proper signature
                concrete_func = model_func.get_concrete_function(
                    tf.TensorSpec(input_shape, tf.float32, name="input")
                )
                
                # Save with explicit signatures to avoid auto-detection issues
                tf.saved_model.save(
                    model, 
                    saved_model_path,
                    signatures={
                        'serving_default': concrete_func
                    }
                )
                
                # Convert SavedModel to ONNX
                onnx_model, _ = tf2onnx.convert.from_saved_model(
                    saved_model_path,
                    input_signature=None,
                    opset=12
                )
                print("✅ SavedModel conversion successful!")
        
        # Save with external data support
        _save_with_external_data(onnx_model, output_path)
        print(f"✅ ONNX model saved to: {output_path}")
            
        return output_path
        
    except ImportError:
        raise ImportError("tf2onnx is required. Install with: pip install tf2onnx")


def export_nemo_to_onnx(model_path: str, 
                        input_sample: Any,
                        output_path: str = "nemo_model.onnx",
                        model_name: str = "nemo_model") -> str:
    """
    Export NeMo model to ONNX format using NVIDIA's official export() method.
    
    Args:
        model_path: Path to saved NeMo model file (.nemo)
        input_sample: Sample input for the model (required)
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to saved ONNX model
    """
    try:
        import onnx
        import os
        
        # Single model case only - simplified
        return _export_single_nemo_model(model_path, input_sample, output_path, model_name)
            
    except ImportError:
        raise ImportError("nemo_toolkit and onnx are required. Install with: pip install nemo_toolkit onnx")


def _export_single_nemo_model(model_path: str, 
                             input_sample: Any,
                             output_path: str,
                             model_name: str) -> str:
    """
    Export a single NeMo model to ONNX format using NVIDIA's official export() method.
    
    Args:
        model_path: Path to saved NeMo model file (.nemo)
        input_sample: Sample input for the model
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to saved ONNX model
    """
    import onnx
    import os
    import tempfile
    
    # Load NeMo model from file
    try:
        from nemo.core.classes import ModelPT
        model = ModelPT.restore_from(model_path)
    except ImportError:
        raise ImportError("nemo_toolkit is required for NeMo model loading")
    
    # Set model to evaluation mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Use NVIDIA's official export() method
    if hasattr(model, 'export'):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Use NeMo's built-in export method with official API
            print(f"🔄 Using NVIDIA's official export() method for {os.path.basename(model_path)}")
            model.export(temp_path, input_example=input_sample, verbose=False)
            
            # Load and re-save with external data support
            onnx_model = onnx.load(temp_path)
            _save_with_external_data(onnx_model, output_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return output_path
            
        except Exception as export_error:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"❌ NVIDIA export() failed: {export_error}")
            
            # Fallback to PyTorch export if model has PyTorch components
            if hasattr(model, 'state_dict'):
                print("🔄 Falling back to PyTorch export...")
                return _export_nemo_via_pytorch_fallback(model, input_sample, output_path, model_name)
            else:
                raise NotImplementedError(f"NeMo model does not support ONNX export: {export_error}")
    else:
        # Fallback to PyTorch export if model has PyTorch components
        if hasattr(model, 'state_dict'):
            print("🔄 Using PyTorch export fallback...")
            return _export_nemo_via_pytorch_fallback(model, input_sample, output_path, model_name)
        else:
            raise NotImplementedError("NeMo model does not support ONNX export")


def _export_nemo_via_pytorch_fallback(model, input_sample, output_path: str, model_name: str) -> str:
    """
    Export NeMo model via PyTorch ONNX export as fallback.
    Note: This is a simplified fallback approach.
    """
    import torch
    import tempfile
    import warnings
    
    warnings.warn("Using PyTorch fallback for NeMo export. Results may vary.")
    
    # Save as PyTorch model temporarily
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
        temp_pytorch_path = temp_file.name
    
    try:
        torch.save(model, temp_pytorch_path)
        
        # Simple PyTorch export without calling external function
        import torch
        import onnx
        import io
        
        # Load model
        pytorch_model = torch.load(temp_pytorch_path, map_location='cpu')
        pytorch_model.eval()
        
        # Convert numpy to tensor if needed
        if isinstance(input_sample, np) and np is not None:
            if hasattr(np, 'ndarray') and isinstance(input_sample, np.ndarray):
                input_sample = torch.from_numpy(input_sample).float()
        
        # Export to ONNX in memory first
        f = io.BytesIO()
        torch.onnx.export(
            pytorch_model,
            input_sample,
            f,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # Load the ONNX model and save with external data support
        f.seek(0)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        _save_with_external_data(onnx_model, output_path)
        
        # Clean up temp file
        if os.path.exists(temp_pytorch_path):
            os.remove(temp_pytorch_path)
            
    except Exception as e:
        print(f"❌ Failed to export PyTorch model: {e}")
        raise


def export_transformers_to_onnx(model_name_or_path: str, 
                               task: str = "text-classification",
                               output_path: str = "transformers_model.onnx",
                               model_name: str = "transformers_model") -> str:
    """
    Convert a Hugging Face Transformers model to ONNX format.
    
    Args:
        model_name_or_path: Model name from Hugging Face Hub or local path
        task: Task type (text-classification, token-classification, etc.)
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to the exported ONNX model
        
    Example:
        export_transformers_to_onnx("bert-base-uncased", "text-classification", "bert.onnx")
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification
        from transformers import AutoTokenizer
        import onnx
        
        print(f"🤗 Converting Transformers model: {model_name_or_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Select appropriate model class based on task
        if task == "text-classification":
            model = ORTModelForSequenceClassification.from_pretrained(model_name_or_path, export=True)
        elif task == "token-classification":
            model = ORTModelForTokenClassification.from_pretrained(model_name_or_path, export=True)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Save the ONNX model
        model.save_pretrained(output_path.replace('.onnx', ''))
        
        # Move the actual ONNX file to the desired location
        onnx_file = os.path.join(output_path.replace('.onnx', ''), "model.onnx")
        if os.path.exists(onnx_file):
            os.rename(onnx_file, output_path)
        
        print(f"✅ Transformers model exported to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to export Transformers model: {e}")
        raise


def export_sklearn_to_onnx(model_path: str,
                          input_sample: Any,
                          output_path: str = "sklearn_model.onnx", 
                          model_name: str = "sklearn_model") -> str:
    """
    Convert a Scikit-learn model to ONNX format.
    
    Args:
        model_path: Path to the pickle file containing the sklearn model
        input_sample: Sample input data for shape inference
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
        
    Returns:
        Path to the exported ONNX model
        
    Example:
        export_sklearn_to_onnx("model.pkl", [[1, 2, 3, 4]], "sklearn_model.onnx")
    """
    try:
        from skl2onnx import to_onnx
        import joblib
        import numpy as np
        
        print(f"🔬 Converting Scikit-learn model: {model_path}")
        
        # Load the model
        if model_path.endswith('.joblib'):
            sklearn_model = joblib.load(model_path)
        else:
            # Assume pickle format
            import pickle
            with open(model_path, 'rb') as f:
                sklearn_model = pickle.load(f)
        
        # Convert input sample to numpy array if needed
        if not isinstance(input_sample, np.ndarray):
            input_sample = np.array(input_sample, dtype=np.float32)
        
        # Convert to ONNX
        onnx_model = to_onnx(sklearn_model, input_sample, target_opset=12)
        
        # Save with external data support
        _save_with_external_data(onnx_model, output_path)
        
        print(f"✅ Scikit-learn model exported to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to export Scikit-learn model: {e}")
        raise


def export_stable_diffusion_to_onnx(model_path: str,
                                   output_path: str = "stable_diffusion.onnx",
                                   model_name: str = "stable_diffusion") -> str:
    """
    Note: Stable Diffusion models are complex pipelines that are not easily 
    converted to a single ONNX model. This function is a placeholder.
    
    For Stable Diffusion, consider using the individual components:
    - Text Encoder
    - UNet
    - VAE Decoder
    
    Each can be converted separately if needed.
    """
    raise NotImplementedError(
        "Stable Diffusion models are complex pipelines that cannot be easily "
        "converted to a single ONNX model. Consider converting individual components "
        "or using TensorRT for acceleration instead."
    )


# Main exports

def export_pytorch_to_onnx(model_path: str,
                           input_sample: Any,
                           output_path: str = "pytorch_model.onnx",
                           model_name: str = "pytorch_model") -> str:
    """
    Export a PyTorch model to ONNX format.
    Args:
        model_path: Path to the PyTorch model file (.pt, .pth)
        input_sample: Sample input tensor (torch.Tensor or numpy.ndarray)
        output_path: Path to save the ONNX model
        model_name: Name for the ONNX model
    Returns:
        Path to the exported ONNX model
    """
    try:
        import torch
        import onnx
        import io
        import numpy as np

        print(f"🔄 Exporting PyTorch model: {model_path}")
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()

        # Convert numpy input to tensor if needed
        if isinstance(input_sample, np.ndarray):
            input_sample = torch.from_numpy(input_sample).float()

        # Export to ONNX in memory
        f = io.BytesIO()
        torch.onnx.export(
            model,
            input_sample,
            f,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        # Load ONNX model and save with external data support
        f.seek(0)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        _save_with_external_data(onnx_model, output_path)

        print(f"✅ PyTorch model exported to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Failed to export PyTorch model: {e}")
        raise

__all__ = [
    'export_tensorflow_to_onnx',
    'export_nemo_to_onnx',
    'export_pytorch_to_onnx',
    'export_transformers_to_onnx',
    'export_sklearn_to_onnx',
    'export_stable_diffusion_to_onnx'
]
