"""
ONNX Export Utilities for AI-Blueprints

Simple API for converting various model formats to ONNX format.
Supports TensorFlow/Keras, NeMo, PyTorch, and Hugging Face Transformers models.
Automatically uses external weights files for models larger than 2GB.

Main Usage (New - accepts loaded models):
    from ais_utils.onnx_export import export_model_to_onnx

    # Automatically detect model type and convert
    # Works with any supported model already loaded in memory
    model = load_your_model()  # Your loaded model
    input_sample = create_sample_input()  # Sample input for the model
    export_model_to_onnx(model, input_sample, output_path="model.onnx")

    # For Transformers models, specify the task
    export_model_to_onnx(transformers_model, None, output_path="bert.onnx", task="text-classification")

Direct Usage (accepts loaded models):
    from ais_utils.onnx_export import (
        export_tensorflow_model_to_onnx,
        export_pytorch_model_to_onnx,
        export_transformers_model_to_onnx,
        export_nemo_model_to_onnx,
        export_sklearn_model_to_onnx
    )

    # Convert loaded models directly
    export_tensorflow_model_to_onnx(tf_model, input_sample, "tf_model.onnx")
    export_pytorch_model_to_onnx(torch_model, input_tensor, "torch_model.onnx")
    export_transformers_model_to_onnx(hf_model, output_path="hf_model.onnx", task="text-classification")
    export_nemo_model_to_onnx(nemo_model, input_sample, "nemo_model.onnx")
    export_sklearn_model_to_onnx(sklearn_model, input_array, "sklearn_model.onnx")

Legacy Usage (from file paths - maintained for compatibility):
    from ais_utils.onnx_export import (
        export_tensorflow_to_onnx,
        export_nemo_to_onnx,
        export_pytorch_to_onnx,
        export_transformers_to_onnx,
        export_sklearn_to_onnx
    )

    # Convert from saved model files
    export_tensorflow_to_onnx("model.keras", input_shape=(1, 28, 28, 1), output_path="model.onnx")
    export_nemo_to_onnx("asr_model.nemo", input_sample={}, output_path="asr_model.onnx")
    export_pytorch_to_onnx("model.pt", input_sample=torch.randn(1, 3, 224, 224), output_path="model.onnx")
    export_transformers_to_onnx("bert-base-uncased", task="text-classification", output_path="bert.onnx")
    export_sklearn_to_onnx("model.pkl", input_sample=[[1, 2, 3, 4]], output_path="sklearn_model.onnx")
"""

import os
import warnings
from typing import Any, Optional, Tuple, Union, List, Dict
from pathlib import Path
import tempfile
import torch
import inspect
import logging

# Optional imports - will be imported when needed
try:
    import numpy as np

    # Use np.ndarray in type hints only if numpy is available
    NDArray = np.ndarray
except ImportError:
    np = None
    # Fallback type for when numpy is not available
    NDArray = Any

logger = logging.getLogger("register_model_logger")


def identify_model_type(model: Any) -> str:
    """
    Automatically identifies the model type based on the loaded object.

    Args:
        model: Model object already loaded in memory

    Returns:
        String identifying the type: 'pytorch', 'tensorflow', 'transformers', 'nemo', 'sklearn'
    """
    model_type_str = str(type(model)).lower()

    # TensorFlow/Keras models (check first to avoid NeMo false positives!)
    try:
        import tensorflow as tf

        if isinstance(model, (tf.keras.Model, tf.Module)):
            return "tensorflow"
        if hasattr(model, "call") and hasattr(model, "trainable_variables"):
            return "tensorflow"
    except ImportError:
        pass

    # NeMo models (check before PyTorch - if has export method, prioritize NeMo!)
    if "nemo" in model_type_str or (
        hasattr(model, "export")
        and "nemo" in getattr(type(model), "__module__", "").lower()
    ):
        return "nemo"

    # PyTorch models (check after NeMo)
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            # Check if it's a Hugging Face Transformers model
            if hasattr(model, "config") and hasattr(model.config, "model_type"):
                return "transformers"
            return "pytorch"
    except ImportError:
        pass

    # Hugging Face Transformers (fallback check)
    if "transformers" in model_type_str or hasattr(model, "generate"):
        return "transformers"

    # Scikit-learn models
    try:
        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            return "sklearn"
    except ImportError:
        pass

    # Check for common ML library patterns
    if hasattr(model, "predict") and hasattr(model, "fit"):
        return "sklearn"

    return "unknown"


def export_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "model.onnx",
    model_name: str = "model",
    task: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Automatically exports any model to ONNX based on detected type.

    Args:
        model: Model object already loaded in memory
        input_sample: Input sample for the model
        output_path: Path to save the ONNX model
        model_name: Model name
        task: Task type (for Transformers models: text-classification, token-classification, translation)
        **kwargs: Additional arguments for specific export methods

    Returns:
        Path to the exported ONNX model
    """
    model_type = identify_model_type(model)
    logger.info(f"🔍 Model identified as: {model_type}")

    if model_type == "pytorch":
        return export_pytorch_model_to_onnx(
            model, input_sample, output_path, model_name, **kwargs
        )

    elif model_type == "tensorflow":
        return export_tensorflow_model_to_onnx(
            model, input_sample, output_path, model_name, **kwargs
        )

    elif model_type == "transformers":
        if task is None:
            task = "text-classification"  # default task
        return export_transformers_model_to_onnx(
            model, input_sample, output_path, model_name, task, **kwargs
        )

    elif model_type == "nemo":
        return export_nemo_model_to_onnx(
            model, input_sample, output_path, model_name, **kwargs
        )

    elif model_type == "sklearn":
        return export_sklearn_model_to_onnx(
            model, input_sample, output_path, model_name, **kwargs
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: pytorch, tensorflow, transformers, nemo, sklearn"
        )


def _save_with_external_data(
    onnx_model: Any, output_path: str, size_threshold: int = 2 * 1024 * 1024 * 1024
) -> None:
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
                convert_attribute=False,
            )
            logger.info(f"Large model saved with external weights: {output_path}")
        else:
            # Save normally for smaller models
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"Model saved to ONNX: {output_path}")

    except ImportError:
        # Fallback to basic save if onnx package not available
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Model saved to ONNX (fallback): {output_path}")


def export_tensorflow_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "tensorflow_model.onnx",
    model_name: str = "tensorflow_model",
    **kwargs,
) -> str:
    """
    Exports a TensorFlow/Keras model already loaded to ONNX format.

    Args:
        model: TensorFlow/Keras model already loaded
        input_sample: Input sample (used to infer input_shape if needed)
        output_path: Path to save the ONNX model
        model_name: Model name
        **kwargs: Additional arguments:
            - opset (int): ONNX opset version (default: 12)
            - use_saved_model (bool): Force SavedModel conversion (default: False)
            - verbose (bool): Enable verbose logging (default: False)

    Returns:
        Path to the saved ONNX model
    """
    try:
        import tf2onnx
        import tensorflow as tf
        import tempfile
        import numpy as np

        # Extract kwargs with defaults
        opset = kwargs.get("opset", 12)
        use_saved_model = kwargs.get("use_saved_model", False)
        verbose = kwargs.get("verbose", False)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(
            f"🔄 Converting loaded TensorFlow/Keras model with opset {opset}..."
        )

        # Infer input shape from sample if needed
        if hasattr(input_sample, "shape"):
            input_shape = input_sample.shape
        elif isinstance(input_sample, (list, tuple)):
            input_shape = (
                (1,) + tuple(np.array(input_sample).shape[1:])
                if len(input_sample) > 0
                else None
            )
        else:
            input_shape = model.input_shape if hasattr(model, "input_shape") else None

        logger.info(f"Converting model with input shape: {input_shape}")

        if use_saved_model:
            logger.info("Using forced SavedModel approach...")
            # Force SavedModel approach
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, "temp_saved_model")

                @tf.function
                def model_func(x):
                    return model(x)

                if input_shape:
                    concrete_func = model_func.get_concrete_function(
                        tf.TensorSpec(input_shape, tf.float32, name="input")
                    )

                    tf.saved_model.save(
                        model,
                        saved_model_path,
                        signatures={"serving_default": concrete_func},
                    )
                else:
                    tf.saved_model.save(model, saved_model_path)

                onnx_model, _ = tf2onnx.convert.from_saved_model(
                    saved_model_path, opset=opset
                )
                logger.info("✅ SavedModel conversion successful!")
        else:
            try:
                # Method 1: Direct Keras model conversion
                logger.info("Trying direct Keras model conversion...")

                # Patch for Keras 3 compatibility
                if not hasattr(model, "output_names"):
                    if hasattr(model, "output"):
                        if isinstance(model.output, list):
                            model.output_names = [
                                f"output_{i}" for i in range(len(model.output))
                            ]
                        else:
                            model.output_names = ["output"]
                    else:
                        model.output_names = ["output"]

                # Create input signature
                if input_shape:
                    spec = tf.TensorSpec(input_shape, tf.float32, name="input")
                    onnx_model, _ = tf2onnx.convert.from_keras(
                        model, input_signature=[spec], opset=opset
                    )
                else:
                    # Try without specific signature
                    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=opset)

                logger.info("✅ Direct conversion successful!")

            except Exception as e:
                logger.info(f"Direct conversion failed: {e}")
                logger.info("Trying SavedModel approach...")

                # Method 2: Fallback via SavedModel
                with tempfile.TemporaryDirectory() as temp_dir:
                    saved_model_path = os.path.join(temp_dir, "temp_saved_model")

                    @tf.function
                    def model_func(x):
                        return model(x)

                    if input_shape:
                        concrete_func = model_func.get_concrete_function(
                            tf.TensorSpec(input_shape, tf.float32, name="input")
                        )

                        tf.saved_model.save(
                            model,
                            saved_model_path,
                            signatures={"serving_default": concrete_func},
                        )
                    else:
                        tf.saved_model.save(model, saved_model_path)

                    onnx_model, _ = tf2onnx.convert.from_saved_model(
                        saved_model_path, opset=opset
                    )
                    logger.info("✅ SavedModel conversion successful!")

        # Save with external data support
        _save_with_external_data(onnx_model, output_path)
        logger.info(f"✅ ONNX model saved to: {output_path}")

        return output_path

    except ImportError:
        raise ImportError("tf2onnx is required. Install with: pip install tf2onnx")


def export_pytorch_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "pytorch_model.onnx",
    model_name: str = "pytorch_model",
    **kwargs,
) -> str:
    """
    Exports a PyTorch model already loaded to ONNX format.

    Args:
        model: PyTorch model already loaded
        input_sample: Example input tensor
        output_path: Path to save the ONNX model
        model_name: Model name
        **kwargs: Additional arguments:
            - input_names (List[str]): Input names (default: ['input'])
            - output_names (List[str]): Output names (default: ['output'])
            - dynamic_axes (Dict): Dynamic axes configuration
            - opset_version (int): ONNX opset version (default: 12)
            - do_constant_folding (bool): Enable constant folding (default: True)
            - export_params (bool): Export model parameters (default: True)
            - verbose (bool): Enable verbose logging (default: False)

    Returns:
        Path to the exported ONNX model
    """
    try:
        import torch
        import onnx
        import io
        import numpy as np

        # Extract kwargs with defaultsonnx
        input_names = kwargs.get("input_names", ["input"])
        output_names = kwargs.get("output_names", ["output"])
        dynamic_axes = kwargs.get("dynamic_axes", None)
        opset_version = kwargs.get("opset", 12)
        do_constant_folding = kwargs.get("do_constant_folding", True)
        export_params = kwargs.get("export_params", True)
        verbose = kwargs.get("verbose", False)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(f"🔄 Exporting loaded PyTorch model with opset {opset_version}...")

        # Ensure model is in evaluation mode
        model.eval()

        # Convert numpy input to tensor if needed
        if isinstance(input_sample, np.ndarray):
            input_sample = torch.from_numpy(input_sample).float()
        try:
            torch.onnx.export(
                model,
                input_sample,
                output_path,
                export_params=export_params,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
                use_external_data_format=True,
            )
        except TypeError as e:
            if "use_external_data_format" in str(e):
                torch.onnx.export(
                    model,
                    input_sample,
                    output_path,
                    export_params=export_params,
                    opset_version=opset_version,
                    do_constant_folding=do_constant_folding,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=verbose,
                )
            else:
                raise

        logger.info(f"✅ PyTorch model exported to: {output_path}")
        return output_path

    except Exception as e:
        logger.info(f"❌ Failed to export PyTorch model: {e}")
        raise


def export_transformers_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "transformers_model.onnx",
    model_name: str = "transformers_model",
    task: str = "text-classification",
    **kwargs,
) -> str:
    """
    Exports a Hugging Face Transformers model already loaded to ONNX format.

    Args:
        model: Transformers model already loaded
        output_path: Path to save the ONNX model
        model_name: Model name
        task: Task type (text-classification, token-classification, translation, seq2seq-lm)
        **kwargs: Additional arguments:
            - opset (int): ONNX opset version (default: 12)
            - feature (str): Feature for transformers export (e.g., "seq2seq-lm")
            - verbose (bool): Enable verbose logging (default: False)

    Returns:
        Path to the exported ONNX model
    """
    try:
        import os
        from pathlib import Path
        from transformers import AutoTokenizer
        from transformers.onnx import export, FeaturesManager
        import torch
        import onnx
        import tempfile
        import io

        # Extract kwargs with defaults
        opset = kwargs.get("opset", 12)
        feature = kwargs.get("feature", None)
        verbose = kwargs.get("verbose", False)
        input_names = kwargs.get("input_names", ["input_ids"])
        output_names = kwargs.get("output_names", ["logits"])
        dynamic_axes = kwargs.get("dynamic_axes", None)
        do_constant_folding = kwargs.get("do_constant_folding", False)

        if isinstance(input_sample, (tuple, list)) and len(input_sample) > 1:
            args_input = input_sample
        else:
            args_input = (input_sample,)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(
            f"🤗 Converting loaded Transformers model for task: {task} with opset {opset}"
        )

        # Try to get the tokenizer from the model
        tokenizer = None
        if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            except:
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            if task in ["text-classification", "token-classification"]:

                try:
                    torch.onnx.export(
                        model,
                        args_input,
                        output_path,
                        export_params=True,
                        opset_version=opset,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        use_external_data_format=True,
                    )
                except TypeError as e:
                    if "use_external_data_format" in str(e):
                        torch.onnx.export(
                            model,
                            args_input,
                            output_path,
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=do_constant_folding,
                            dynamic_axes=dynamic_axes,
                            input_names=input_names,
                            output_names=output_names,
                            verbose=verbose,
                        )
                    else:
                        raise

            elif task in ["translation", "seq2seq-lm"]:
                # Use feature if provided, otherwise infer from task
                feature_name = feature or "seq2seq-lm"

                try:
                    # Para tradução, usar a API oficial do Transformers
                    model_kind, model_onnx_config = (
                        FeaturesManager.check_supported_model_or_raise(
                            model, feature=feature_name
                        )
                    )
                    onnx_config = model_onnx_config(model.config)

                    export(
                        preprocessor=tokenizer,
                        model=model,
                        config=onnx_config,
                        opset=opset,
                        output=Path(output_path),
                    )
                except Exception as e:
                    logger.warning(
                        f"Official export failed: {e}, trying PyTorch fallback"
                    )

                    try:
                        torch.onnx.export(
                            model,
                            args_input,
                            output_path,
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=do_constant_folding,
                            dynamic_axes=dynamic_axes,
                            input_names=input_names,
                            output_names=output_names,
                            verbose=verbose,
                            use_external_data_format=True,
                        )
                    except TypeError as e:
                        if "use_external_data_format" in str(e):
                            torch.onnx.export(
                                model,
                                args_input,
                                output_path,
                                export_params=True,
                                opset_version=opset,
                                do_constant_folding=do_constant_folding,
                                dynamic_axes=dynamic_axes,
                                input_names=input_names,
                                output_names=output_names,
                                verbose=verbose,
                            )
                        else:
                            raise

            else:
                try:
                    torch.onnx.export(
                        model,
                        args_input,
                        output_path,
                        export_params=True,
                        opset_version=opset,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        use_external_data_format=True,
                    )
                except TypeError as e:
                    if "use_external_data_format" in str(e):
                        torch.onnx.export(
                            model,
                            args_input,
                            output_path,
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=do_constant_folding,
                            dynamic_axes=dynamic_axes,
                            input_names=input_names,
                            output_names=output_names,
                            verbose=verbose,
                        )
                    else:
                        raise

        logger.info(f"✅ Transformers model exported to: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.info(f"❌ Failed to export Transformers model: {e}")
        raise


def export_nemo_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "nemo_model.onnx",
    model_name: str = "nemo_model",
    **kwargs,
) -> str:
    """
    Exports a NeMo model already loaded to ONNX format.

    Args:
        model: NeMo model already loaded
        input_sample: Input sample for the model
        output_path: Path to save the ONNX model
        model_name: Model name
        **kwargs: Additional arguments:
            - export_format (str): NeMo export format (default: "ONNX")
            - check_trace (bool): Check trace for NeMo models (default: True)
            - verbose (bool): Enable verbose logging (default: False)
            - input_names (List[str]): Input names for fallback PyTorch export
            - output_names (List[str]): Output names for fallback PyTorch export
            - dynamic_axes (Dict): Dynamic axes for fallback PyTorch export

    Returns:
        Path to the saved ONNX model
    """
    try:
        import onnx
        import os
        import tempfile
        import inspect

        # Extract kwargs with defaults
        export_format = kwargs.get("export_format", "ONNX")
        check_trace = kwargs.get("check_trace", False)
        verbose = kwargs.get("verbose", False)
        input_names = kwargs.get("input_names", None)
        output_names = kwargs.get("output_names", None)
        dynamic_axes = kwargs.get("dynamic_axes", None)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(f"🔄 Exporting loaded NeMo model with format: {export_format}")

        # Set model to evaluation mode
        if hasattr(model, "eval"):
            model.eval()

        # Use official NVIDIA export() method if available
        if hasattr(model, "export"):
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                logger.info(f"🔄 Using official NVIDIA export() method")

                model.export(
                    temp_path,
                    input_example=input_sample,
                    verbose=verbose,
                    check_trace=check_trace,
                )

                # Load and re-save with external data support
                onnx_model = onnx.load(temp_path)
                _save_with_external_data(onnx_model, output_path)

                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                return output_path

            except Exception as export_error:
                # Clean up temporary file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                logger.info(f"❌ NVIDIA export() failed: {export_error}")

        # Fallback to PyTorch export if model has PyTorch components
        if hasattr(model, "state_dict"):
            logger.info("🔄 Using PyTorch export fallback...")

            return export_pytorch_model_to_onnx(
                model=model,
                input_sample=input_sample,
                output_path=output_path,
                model_name=model_name,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
            )
        else:
            raise NotImplementedError("NeMo model does not support ONNX export")

    except ImportError:
        raise ImportError(
            "nemo_toolkit and onnx are required. Install with: pip install nemo_toolkit onnx"
        )


def export_sklearn_model_to_onnx(
    model: Any,
    input_sample: Any,
    output_path: str = "sklearn_model.onnx",
    model_name: str = "sklearn_model",
    **kwargs,
) -> str:
    """
    Exports a Scikit-learn model already loaded to ONNX format.

    Args:
        model: Sklearn model already loaded
        input_sample: Example input data for shape inference
        output_path: Path to save the ONNX model
        model_name: Model name
        **kwargs: Additional arguments:
            - target_opset (int): Target ONNX opset version (default: 12)
            - initial_types (List): Initial types for sklearn conversion
            - verbose (bool): Enable verbose logging (default: False)

    Returns:
        Path to the exported ONNX model
    """
    try:
        from skl2onnx import to_onnx
        import numpy as np

        # Extract kwargs with defaults
        target_opset = kwargs.get("opset", 12)
        initial_types = kwargs.get("initial_types", None)
        verbose = kwargs.get("verbose", False)

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(
            f"🔬 Converting loaded Scikit-learn model with opset {target_opset}..."
        )

        # Convert input sample to numpy array if needed
        if not isinstance(input_sample, np.ndarray):
            input_sample = np.array(input_sample, dtype=np.float32)

        # Build conversion arguments
        convert_kwargs = {"target_opset": target_opset}

        if initial_types is not None:
            convert_kwargs["initial_types"] = initial_types

        # Convert to ONNX
        onnx_model = to_onnx(model, input_sample, **convert_kwargs)

        # Save with external data support
        _save_with_external_data(onnx_model, output_path)

        logger.info(f"✅ Scikit-learn model exported to: {output_path}")
        return output_path

    except Exception as e:
        logger.info(f"❌ Failed to export Scikit-learn model: {e}")
        raise


__all__ = [
    "export_model_to_onnx",
    "identify_model_type",
    "export_tensorflow_model_to_onnx",
    "export_pytorch_model_to_onnx",
    "export_transformers_model_to_onnx",
    "export_nemo_model_to_onnx",
    "export_sklearn_model_to_onnx",
]
