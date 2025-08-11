"""
AI-Blueprints MLflow Utilities with ONNX Integration for In-Memory Models

Provides enhanced MLflow logging capabilities with automatic ONNX export support
for models already loaded in memory. Supports individual model directory generation 
for ML pipelines with maximum efficiency.

Uses ModelExportConfig for clean, per-model configuration with manual object creation.

Supported Model Types (In-Memory Only):
- 🎙️ NeMo: Loaded NeMo models (ASR, TTS, Translation, etc.)
- 🧠 TensorFlow/Keras: Loaded tf.keras.Model objects
- 🤖 PyTorch: Loaded torch.nn.Module objects
- 🔤 Transformers: Loaded Hugging Face transformer models
- 📊 Scikit-learn: Loaded sklearn models

Generated Artifacts:
- ONNX versions (.onnx) with naming: {model_name}.onnx
- External weights (.onnx.data) for models >2GB
- Individual model directories ready for deployment
- MLflow artifacts and signatures

Key Features:
- ✅ Manual ModelExportConfig object creation
- 🚀 Works ONLY with pre-loaded models (maximum efficiency!)
- ✅ Automatic ONNX conversion from multiple model formats
- ✅ Individual model directories for each model
- ✅ Support for both single and multiple model pipelines
- ✅ Ready for deployment and easy model management
- ✅ Type safety and documentation
- ✅ Extensible for new model types

API Examples:

    from ais_utils.utils import ModelExportConfig, log_model
    
    # Your models already loaded in memory
    encoder_model = tf.keras.models.load_model("encoder.keras")
    decoder_model = nemo_asr.models.EncDecCTCModel.restore_from("decoder.nemo")
    
    # Create configs manually
    model_configs = [
        ModelExportConfig(
            model=encoder_model,
            model_name="bert_encoder",
            input_sample=sample_input,
            create_triton_structure=True
        ),
        ModelExportConfig(
            model=decoder_model,
            model_name="bert_decoder", 
            input_sample=sample_audio,
            create_triton_structure=True
        )
    ]
    
    log_model(
        artifact_path="bert_pipeline",
        python_model=BERTTourismModel(),
        artifacts={"corpus": "corpus.csv", "embeddings": "embeddings.csv"},
        models_to_convert_onnx=model_configs
    )

Benefits:
- 🚀 Maximum efficiency: No file I/O overhead
- 💾 Reduced memory usage: No model duplication
- 🛡️ Simplified error handling: No file path issues
- 🧹 Cleaner code: Single API approach
- ⚡ Faster execution: Direct memory to ONNX conversion
- ✅ Manual control over model configuration
- ✅ Individual model directories as MLflow artifacts (Triton-style)
- ✅ Direct ONNX file storage option (simple file copy)
- ✅ Per-model input shapes and samples
- ✅ Flexible structure creation (Triton directories or direct files)
"""

import os
import tempfile
import shutil
from typing import Any, Dict, Optional, Union, Tuple, List
from pathlib import Path
import logging
from dataclasses import dataclass, field

# Optional imports
try:
    import yaml
except ImportError:
    yaml = None

try:
    import onnx
    import numpy as np
except ImportError:
    onnx = None
    np = None

# Set up logging
logger = logging.getLogger("register_model_logger")

@dataclass
class ModelExportConfig:
    """
    Configuration for ONNX model export from loaded models.
    
    All model-specific parameters should be passed as kwargs when creating the config.
    This keeps the interface clean and allows each export method to receive 
    only the parameters it needs.
    """
    model: Any                      # Model already loaded in memory (required)
    model_name: str                 # Model name for file/directory
    input_shape: Optional[Tuple] = None
    input_sample: Optional[Any] = None
    model_type: Optional[str] = None  # Auto-detect if None
    task: str = "text-classification"  # for Transformers models
    create_triton_structure: bool = False  # Create Triton-style directories or just save ONNX file
    
    def __init__(self, model: Any, model_name: str, **kwargs):
        """
        Initialize ModelExportConfig with model-specific parameters as kwargs.
        
        Args:
            model: Model already loaded in memory
            model_name: Model name for file/directory
            **kwargs: All other parameters including model-specific ones
        """
        self.model = model
        self.model_name = model_name
        
        # Extract common parameters
        self.input_shape = kwargs.pop('input_shape', None)
        self.input_sample = kwargs.pop('input_sample', None)
        self.model_type = kwargs.pop('model_type', None)
        self.task = kwargs.pop('task', 'text-classification')
        self.create_triton_structure = kwargs.pop('create_triton_structure', False)
        
        # Store all remaining kwargs for the export methods
        self.export_kwargs = kwargs
        
        # Auto-detect model type if not provided
        if self.model_type is None:
            from onnx_export import identify_model_type
            self.model_type = identify_model_type(self.model)
    
    def get_onnx_filename(self) -> str:
        """Return ONNX filename."""
        return f"{self.model_name}.onnx"
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.model is None:
            raise ValueError("Model object must be provided (no file paths supported)")
            
        if self.model_type in ['nemo', 'pytorch', 'sklearn'] and self.input_sample is None:
            raise ValueError(f"Input sample required for {self.model_type} model '{self.model_name}'.")
        
        if self.model_type == 'tensorflow' and self.input_shape is None and self.input_sample is None:
            raise ValueError(f"Input shape or input sample required for TensorFlow model '{self.model_name}'.")
    
    def __post_init__(self):
        """Auto-detect model type from loaded model"""
        if self.model_type is None:
            from onnx_export import identify_model_type
            self.model_type = identify_model_type(self.model)
    
    def get_onnx_filename(self) -> str:
        """Return ONNX filename."""
        return f"{self.model_name}.onnx"
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.model is None:
            raise ValueError("Model object must be provided (no file paths supported)")
            
        if self.model_type in ['nemo', 'pytorch', 'sklearn'] and self.input_sample is None:
            raise ValueError(f"Input sample required for {self.model_type} model '{self.model_name}'.")
        
        if self.model_type == 'tensorflow' and self.input_shape is None and self.input_sample is None:
            raise ValueError(f"Input shape or input sample required for TensorFlow model '{self.model_name}'.")

def create_model_export_configs(
    models_dict: Dict[str, Any],
    model_names: Dict[str, str],
    input_shapes: Optional[Dict[str, Tuple]] = None,
    input_samples: Optional[Dict[str, Any]] = None,
    create_triton_structure: bool = True,
    tasks: Optional[Dict[str, str]] = None
) -> List[ModelExportConfig]:
    """
    Helper function to create ModelExportConfig objects from loaded models.
    
    RECOMMENDATION: Use manual ModelExportConfig creation for better control:
    
        # Manual creation (preferred):
        configs = [
            ModelExportConfig(
                model=encoder_model,
                model_name="bert_encoder",
                input_sample=sample_input,
                create_triton_structure=True
            ),
            ModelExportConfig(
                model=decoder_model,
                model_name="bert_decoder", 
                input_sample=sample_audio,
                create_triton_structure=True
            )
        ]
    
    Args:
        models_dict: Dictionary mapping model_key to loaded model object
        model_names: Dictionary mapping model_key to model_name (for file naming)
        input_shapes: Optional dictionary mapping model_key to input_shape (for TensorFlow models)
        input_samples: Optional dictionary mapping model_key to input_sample (for NeMo/PyTorch/sklearn)
        create_triton_structure: Whether to create Triton-style directories for all models
        tasks: Optional dictionary mapping model_key to task (for Transformers models)
        
    Returns:
        List of ModelExportConfig objects
    """
    configs = []
    
    for model_key, model_obj in models_dict.items():
        if model_key not in model_names:
            raise ValueError(f"Model name not provided for model_key: {model_key}")
            
        config = ModelExportConfig(
            model=model_obj,
            model_name=model_names[model_key],
            input_shape=input_shapes.get(model_key) if input_shapes else None,
            input_sample=input_samples.get(model_key) if input_samples else None,
            task=tasks.get(model_key, "text-classification") if tasks else "text-classification",
            create_triton_structure=create_triton_structure
        )
        configs.append(config)
    
    return configs


def _convert_single_model_to_onnx(config: ModelExportConfig) -> str:
    """
    Convert a single model to ONNX format with model-specific parameters.
    
    Args:
        config: Model export configuration with all parameters
        output_path: Where to save the ONNX file
        
    Returns:
        Path to the exported ONNX model
    """
    try:
        from onnx_export import export_model_to_onnx
        onnx_filename = config.get_onnx_filename()
        
        logger.info(f"🔄 Converting {config.model_type} model: {config.model_name}")
        
        # Use the main API with all kwargs passed through
        return export_model_to_onnx(
            model=config.model,
            input_sample=config.input_sample,
            output_path=onnx_filename,
            model_name=config.model_name,
            task=config.task,
            **config.export_kwargs  # Pass all kwargs directly
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {config.model_name}: {e}")
        raise


def _generate_onnx_from_models(model_configs: List[ModelExportConfig]) -> Union[str, Dict[str, str]]:
    """
    Generate ONNX models from list of ModelExportConfig.
    
    Args:
        model_configs: List of ModelExportConfig objects
        
    Returns:
        String path for single model or Dict of {model_key: onnx_path} for multiple
    """
    if not model_configs:
        return None
    
    onnx_results = {}
    
    for config in model_configs:
        try:
            onnx_path = _convert_single_model_to_onnx(config)
            onnx_results[config.model_name] = onnx_path
            logger.info(f"✅ Converted {config.model_name}: {onnx_path}")
        except Exception as e:
            logger.error(f"❌ Failed to convert {config.model_name}: {e}")
            continue
    
    if len(onnx_results) == 1:
        return list(onnx_results.values())[0]
    elif len(onnx_results) > 1:
        return onnx_results
    else:
        return None





def _create_model_directories(onnx_models: Union[str, Dict[str, str]], 
                            create_triton_structure: bool = False) -> Dict[str, str]:
    """
    Create individual model directories with ONNX models for MLflow artifacts.
    
    Args:
        onnx_models: Single ONNX path or dict of {model_name: onnx_path}
        create_triton_structure: If True, creates Triton-style directories ({model_name}/1/model.onnx).
                                If False, just copies the ONNX file with the model_name as filename.
        
    Returns:
        Dictionary mapping model_names to their directory paths or file paths
        
    Generated Structure (create_triton_structure=True):
        {model_name}/
        └── 1/
            └── model.onnx
            
    Generated Structure (create_triton_structure=False):
        {model_name}.onnx
    """
    
    model_paths = {}
    
    # Handle single model or multiple models
    models_to_process = {}
    if isinstance(onnx_models, str):
        # Single model - use filename as model_name
        model_name = Path(onnx_models).stem
        models_to_process[model_name] = onnx_models
    elif isinstance(onnx_models, dict):
        models_to_process = onnx_models
    
    if create_triton_structure:
        logger.info(f"  Creating Triton model directories for {len(models_to_process)} models: {list(models_to_process.keys())}")
    else:
        logger.info(f"  Copying ONNX models for {len(models_to_process)} models: {list(models_to_process.keys())}")
    
    for model_name, onnx_path in models_to_process.items():
        if not os.path.exists(onnx_path):
            logger.warning(f"⚠️ Model file not found: {onnx_path}")
            continue
        
        try:
            if create_triton_structure:
                # Create Triton-style model directory structure
                model_dir = model_name
                version_dir = os.path.join(model_dir, "1")  # Always use version 1
                
                os.makedirs(version_dir, exist_ok=True)
                
                # Detect model file type and copy with appropriate name
                original_path = Path(onnx_path)
                if original_path.suffix == '.onnx':
                    model_dest = os.path.join(version_dir, "model.onnx")
                elif original_path.suffix in ['.pt', '.pth']:
                    model_dest = os.path.join(version_dir, "model.pt")
                elif original_path.suffix in ['.keras', '.h5']:
                    model_dest = os.path.join(version_dir, "model.savedmodel")  # TensorFlow format
                else:
                    # Default to ONNX naming for other formats
                    model_dest = os.path.join(version_dir, "model.onnx")
                
                shutil.copy2(onnx_path, model_dest)
                logger.info(f"📁 Created Triton model directory: {model_name}/1/{Path(model_dest).name}")
                
                # Copy external data file if it exists (for ONNX models >2GB)
                onnx_data_path = f"{onnx_path}.data"
                if os.path.exists(onnx_data_path):
                    data_dest = os.path.join(version_dir, f"{Path(model_dest).name}.data")
                    shutil.copy2(onnx_data_path, data_dest)
                    logger.info(f"📁 Added model data: {model_name}/1/{Path(model_dest).name}.data")
                
                model_paths[model_name] = model_dir
                
            else:
                # Just copy the ONNX file with model_name as filename
                original_path = Path(onnx_path)
                if original_path.suffix == '.onnx':
                    model_dest = f"{model_name}.onnx"
                elif original_path.suffix in ['.pt', '.pth']:
                    model_dest = f"{model_name}.pt"
                elif original_path.suffix in ['.keras', '.h5']:
                    model_dest = f"{model_name}.keras"
                else:
                    # Default to ONNX extension
                    model_dest = f"{model_name}.onnx"
                
                # Check if source and destination are the same file
                if os.path.abspath(onnx_path) == os.path.abspath(model_dest):
                    logger.info(f"📄 Model file already has correct name: {model_dest}")
                    model_paths[model_name] = model_dest
                else:
                    shutil.copy2(onnx_path, model_dest)
                    logger.info(f"📄 Copied model file: {model_dest}")
                    model_paths[model_name] = model_dest
                
                # Copy external data file if it exists (for ONNX models >2GB)
                onnx_data_path = f"{onnx_path}.data"
                if os.path.exists(onnx_data_path):
                    data_dest = f"{model_dest}.data"
                    # Check if data files are the same
                    if os.path.abspath(onnx_data_path) != os.path.abspath(data_dest):
                        shutil.copy2(onnx_data_path, data_dest)
                        logger.info(f"📄 Added model data: {data_dest}")
                    else:
                        logger.info(f"📄 Model data file already has correct name: {data_dest}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process model {model_name}: {e}")
            continue
    
    return model_paths


def log_model(artifact_path: str,
              python_model: Optional[Any] = None,
              artifacts: Optional[Dict[str, str]] = None,
              conda_env: Optional[Union[str, Dict]] = None,
              code_paths: Optional[List[str]] = None,
              signature: Optional[Any] = None,
              input_example: Optional[Any] = None,
              pip_requirements: Optional[Union[List[str], str]] = None,
              extra_pip_requirements: Optional[Union[List[str], str]] = None,
              metadata: Optional[Dict[str, Any]] = None,
              models_to_convert_onnx: Optional[List[ModelExportConfig]] = None,
              **kwargs) -> None:
    """
    Enhanced MLflow log_model with automatic ONNX export and model directory creation.
    
    Uses ModelExportConfig for clean, per-model configuration.
    
    Args:
        artifact_path: Run-relative artifact path to log the model as
        python_model: PyFunc-compatible model instance
        artifacts: Dictionary of model artifacts {name: path}
        conda_env: Conda environment specification
        code_paths: List of code paths to include
        signature: Model signature (auto-inferred from input_example if not provided)
        input_example: Example model input for signature inference
        pip_requirements: List of pip requirements or path to requirements file
        extra_pip_requirements: Additional pip requirements
        metadata: Metadata dictionary to log with the model
        models_to_convert_onnx: List[ModelExportConfig] with configuration for each model
        **kwargs: Additional arguments passed to mlflow.pyfunc.log_model
        
    Example (ONNX conversion + Model directories):
        model_configs = [
            ModelExportConfig(
                model_path="encoder.keras",
                triton_model_name="densenet_onnx",
                input_shape=(1, 224, 224, 3)
            ),
            ModelExportConfig(
                model_path="decoder.pt",
                triton_model_name="bert_pytorch",
                input_sample=sample_tensor
            ),
        ]
        
        log_model(
            artifact_path="my_pipeline",
            python_model=MyPipelineModel(),
            models_to_convert_onnx=model_configs
        )
        
        # This will create individual model directories as artifacts:
        # - model_densenet_onnx/1/model.onnx
        # - model_bert_pytorch/1/model.pt
    """
    # Import MLflow components
    try:
        import mlflow
        import mlflow.pyfunc
    except ImportError:
        raise ImportError("MLflow is required but not installed. Install with: pip install mlflow")
        
    # Prepare artifacts dictionary
    final_artifacts = artifacts.copy() if artifacts else {}
    
    # Generate ONNX model(s) if specified
    onnx_result = None
    triton_paths = None
    
    if models_to_convert_onnx:
        logger.info("🔧 Generating ONNX model(s) for specified models...")
        
        onnx_result = _generate_onnx_from_models(
            model_configs=models_to_convert_onnx
        )
        
        # Add ONNX models as individual artifacts
        if onnx_result:
            if isinstance(onnx_result, dict):
                # Multiple models - add each as individual artifact
                for model_name, onnx_path in onnx_result.items():
                    artifact_key = f"onnx_{model_name}"
                    final_artifacts[artifact_key] = onnx_path
                    logger.info(f"📦 Added ONNX artifact: {artifact_key} -> {onnx_path}")
            elif isinstance(onnx_result, str):
                # Single model
                final_artifacts["onnx_model"] = onnx_result
                logger.info(f"📦 Added ONNX artifact: onnx_model -> {onnx_result}")
      
        # Create individual model directories for artifacts
        if onnx_result:
            logger.info("  Creating individual model directories for artifacts...")
            
            try:
                # Check if any model config requires Triton structure
                create_triton = any(config.create_triton_structure for config in models_to_convert_onnx)
                
                model_paths = _create_model_directories(
                    onnx_models=onnx_result,
                    create_triton_structure=create_triton
                )
                
                # Add each model directory/file as a separate artifact
                if model_paths:
                    for model_name, model_path in model_paths.items():
                        if create_triton:
                            artifact_key = f"model_{model_name}"
                            logger.info(f"📦 Added Triton model artifact: {artifact_key} -> {model_path}")
                        else:
                            artifact_key = f"model_{model_name}"
                            logger.info(f"📦 Added model file artifact: {artifact_key} -> {model_path}")
                        final_artifacts[artifact_key] = model_path
                    
                    if create_triton:
                        logger.info(f"✅ Created {len(model_paths)} Triton model directories!")
                    else:
                        logger.info(f"✅ Copied {len(model_paths)} model files!")
                
            except Exception as e:
                logger.error(f"❌ Failed to create model directories: {e}")
                # Note: ONNX artifacts are already added above as fallback
    
    try:
        # MLflow logging with ONNX and Triton artifacts included
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=python_model,
            artifacts=final_artifacts,
            conda_env=conda_env,
            code_paths=code_paths,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
            **kwargs
        )
        
        # Log artifacts information
        artifact_list = list(final_artifacts.keys()) if final_artifacts else []
        logger.info(f"Model logged with artifacts: {artifact_list}")
        
        if models_to_convert_onnx and onnx_result:
            if model_paths:
                logger.info(f"✅ Model logged with {len(model_paths)} model directories created!")
            else:
                # Log ONNX success without model directories
                if isinstance(onnx_result, dict):
                    logger.info(f"✅ Model logged with {len(onnx_result)} ONNX conversions!")
                else:
                    logger.info("✅ Model logged with ONNX conversion!")
        
        if not models_to_convert_onnx:
            logger.info("✅ Model logged successfully!")
            
    except Exception as e:
        logger.error(f"Error logging model: {e}")
        raise
    finally:
        # Clean up temporary ONNX files when model directories were created
        if onnx_result and model_paths:
            try:
                if isinstance(onnx_result, dict):
                    # Multiple models cleanup
                    for model_name, onnx_path in onnx_result.items():
                        if os.path.exists(onnx_path):
                            os.remove(onnx_path)
                            logger.debug(f"🧹 Cleaned up temporary ONNX: {onnx_path}")
                        if os.path.exists(f"{onnx_path}.data"):
                            os.remove(f"{onnx_path}.data")
                            logger.debug(f"🧹 Cleaned up temporary ONNX data: {onnx_path}.data")
                elif isinstance(onnx_result, str):
                    # Single model cleanup
                    if os.path.exists(onnx_result):
                        os.remove(onnx_result)
                        logger.debug(f"🧹 Cleaned up temporary ONNX: {onnx_result}")
                    if os.path.exists(f"{onnx_result}.data"):
                        os.remove(f"{onnx_result}.data")
                        logger.debug(f"🧹 Cleaned up temporary ONNX data: {onnx_result}.data")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary ONNX files: {cleanup_error}")


def create_model_directories_standalone(onnx_models: Union[str, Dict[str, str], List[str]],
                                     output_dir: Optional[str] = None,
                                     create_triton_structure: bool = True) -> Dict[str, str]:
    """
    Create individual model directories from existing ONNX models.
    
    This is a standalone function that can be used independently of MLflow logging.
    
    Args:
        onnx_models: Single ONNX path, dict of {model_name: onnx_path}, or list of ONNX paths
        output_dir: Optional output directory (if None, creates directories in current working directory)
        create_triton_structure: If True, creates Triton-style directories. If False, just copies files.
        
    Returns:
        Dictionary mapping model names to their directory/file paths
        
    Example:
        # Single model with Triton structure
        model_paths = create_model_directories_standalone("my_model.onnx", create_triton_structure=True)
        
        # Multiple models without Triton structure (just copy files)
        models = {"densenet_onnx": "encoder.onnx", "bert_pytorch": "decoder.pt"}
        model_paths = create_model_directories_standalone(models, create_triton_structure=False)
        
        # Result structure (create_triton_structure=True):
        # densenet_onnx/
        # └── 1/
        #     └── model.onnx
        # bert_pytorch/
        # └── 1/
        #     └── model.pt
        
        # Result structure (create_triton_structure=False):
        # densenet_onnx.onnx
        # bert_pytorch.pt
    """
    # Convert different input formats to consistent dict format
    if isinstance(onnx_models, str):
        # Single model path
        model_name = Path(onnx_models).stem
        models_dict = {model_name: onnx_models}
    elif isinstance(onnx_models, list):
        # List of model paths
        models_dict = {}
        for path in onnx_models:
            model_name = Path(path).stem
            models_dict[model_name] = path
    elif isinstance(onnx_models, dict):
        # Already in correct format
        models_dict = onnx_models
    else:
        raise ValueError(f"Unsupported onnx_models type: {type(onnx_models)}")
    
    if create_triton_structure:
        logger.info(f"  Creating Triton model directories for {len(models_dict)} models")
    else:
        logger.info(f"  Copying model files for {len(models_dict)} models")
    
    # Change working directory if output_dir is specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    
    try:
        return _create_model_directories(
            onnx_models=models_dict,
            create_triton_structure=create_triton_structure
        )
    finally:
        # Restore original working directory
        if original_cwd:
            os.chdir(original_cwd)


