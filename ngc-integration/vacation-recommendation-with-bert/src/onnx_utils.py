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
        return "model.onnx"
    
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
        
    Returns:
        Path to the model directory containing the ONNX model and other files
    """
    try:
        from onnx_export import export_model_to_onnx
        
        # Create model directory
        model_dir = config.model_name
        os.makedirs(model_dir, exist_ok=True)
        
        # ONNX file path inside the model directory
        onnx_path = os.path.join(model_dir, config.get_onnx_filename())
        
        logger.info(f"🔄 Converting {config.model_type} model: {config.model_name}")
        logger.info(f"📁 Model directory: {model_dir}")
        
        # Use the main API with all kwargs passed through
        export_model_to_onnx(
            model=config.model,
            input_sample=config.input_sample,
            output_path=onnx_path,
            model_name=config.model_name,
            task=config.task,
            **config.export_kwargs  # Pass all kwargs directly
        )
        
        # Return the model directory path (not the ONNX file path)
        return model_dir
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {config.model_name}: {e}")
        raise


def _generate_onnx_from_models(model_configs: List[ModelExportConfig]) -> Union[str, Dict[str, str]]:
    """
    Generate ONNX models from list of ModelExportConfig.
    
    Args:
        model_configs: List of ModelExportConfig objects
        
    Returns:
        String path for single model directory or Dict of {model_key: model_dir_path} for multiple
    """
    if not model_configs:
        return None
    
    model_results = {}
    
    for config in model_configs:
        try:
            model_dir = _convert_single_model_to_onnx(config)
            model_results[config.model_name] = model_dir
            logger.info(f"✅ Converted {config.model_name} to directory: {model_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to convert {config.model_name}: {e}")
            continue
    
    if len(model_results) == 1:
        return list(model_results.values())[0]
    elif len(model_results) > 1:
        return model_results
    else:
        return None





def _create_model_directories(model_dirs: Union[str, Dict[str, str]], 
                            create_triton_structure: bool = False) -> Dict[str, str]:
    """
    Create individual model directories with ONNX models for MLflow artifacts.
    
    Args:
        model_dirs: Single model directory path or dict of {model_name: model_dir_path}
        create_triton_structure: If True, creates Triton-style directories ({model_name}/1/).
                                If False, uses the model directories as-is.
        
    Returns:
        Dictionary mapping model_names to their directory paths
        
    Generated Structure (create_triton_structure=True):
        {model_name}/
        └── 1/
            └── model.onnx
            └── [all other files from source model directory]
            
    Generated Structure (create_triton_structure=False):
        {model_name}/
        └── model.onnx
        └── [all other files from source model directory]
    """
    
    model_paths = {}
    
    # Handle single model or multiple models
    models_to_process = {}
    if isinstance(model_dirs, str):
        # Single model - use directory name as model_name
        model_name = Path(model_dirs).name
        models_to_process[model_name] = model_dirs
    elif isinstance(model_dirs, dict):
        models_to_process = model_dirs
    
    if create_triton_structure:
        logger.info(f"  Creating Triton model directories for {len(models_to_process)} models: {list(models_to_process.keys())}")
    else:
        logger.info(f"  Using model directories for {len(models_to_process)} models: {list(models_to_process.keys())}")
    
    for model_name, source_dir in models_to_process.items():
        if not os.path.exists(source_dir):
            logger.warning(f"⚠️ Model directory not found: {source_dir}")
            continue
        
        try:
            if create_triton_structure:
                # Create Triton-style model directory structure
                target_model_dir = f"{model_name}_triton"
                version_dir = os.path.join(target_model_dir, "1")  # Always use version 1
                
                os.makedirs(version_dir, exist_ok=True)
                
                # Copy all contents from source directory to version directory
                source_path = Path(source_dir)
                if source_path.is_dir():
                    logger.info(f"📁 Copying all contents from {source_dir} to {version_dir}")
                    
                    # Copy all files and subdirectories from source to version directory
                    for item in source_path.iterdir():
                        dest_path = os.path.join(version_dir, item.name)
                        if item.is_file():
                            shutil.copy2(str(item), dest_path)
                            logger.debug(f"  📄 Copied file: {item.name}")
                        elif item.is_dir():
                            shutil.copytree(str(item), dest_path, dirs_exist_ok=True)
                            logger.debug(f"  📁 Copied directory: {item.name}")
                    
                    # Ensure there's a model.onnx file in the version directory
                    model_onnx_path = os.path.join(version_dir, "model.onnx")
                    if not os.path.exists(model_onnx_path):
                        # Look for any .onnx file in the version directory and rename it to model.onnx
                        onnx_files = [f for f in os.listdir(version_dir) if f.endswith('.onnx')]
                        if onnx_files:
                            original_onnx = os.path.join(version_dir, onnx_files[0])
                            os.rename(original_onnx, model_onnx_path)
                            logger.info(f"📄 Renamed {onnx_files[0]} to model.onnx")
                        else:
                            logger.warning(f"⚠️ No ONNX file found in directory {source_dir}")
                    
                    logger.info(f"� Created Triton model directory: {target_model_dir}/1/")
                    model_paths[model_name] = target_model_dir
                else:
                    logger.error(f"❌ Source is not a directory: {source_dir}")
                    continue
                
            else:
                # Use the source directory as-is (already contains model files)
                source_path = Path(source_dir)
                if source_path.is_dir():
                    # Just return the existing directory path
                    logger.info(f"📁 Using existing model directory: {source_dir}")
                    model_paths[model_name] = source_dir
                else:
                    logger.error(f"❌ Source is not a directory: {source_dir}")
                    continue
                
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
    model_result = None
    
    if models_to_convert_onnx:
        logger.info("🔧 Generating ONNX model(s) for specified models...")
        
        model_result = _generate_onnx_from_models(
            model_configs=models_to_convert_onnx
        )
        
        # Add model directories as individual artifacts (these now contain ONNX + other files)
        if model_result:
            if isinstance(model_result, dict):
                # Multiple models - add each model directory as individual artifact
                for model_name, model_dir in model_result.items():
                    artifact_key = f"model_{model_name}"
                    final_artifacts[artifact_key] = model_dir
                    logger.info(f"📦 Added model directory artifact: {artifact_key} -> {model_dir}")
            elif isinstance(model_result, str):
                # Single model directory
                final_artifacts["model_directory"] = model_result
                logger.info(f"📦 Added model directory artifact: model_directory -> {model_result}")
      
        # Create Triton-style directories if needed
        if model_result:
            
            try:
                # Check if any model config requires Triton structure
                create_triton = any(config.create_triton_structure for config in models_to_convert_onnx)
                
                if create_triton:
                    triton_paths = _create_model_directories(
                        model_dirs=model_result,
                        create_triton_structure=True
                    )
                    
                    # Add each Triton model directory as a separate artifact
                    if triton_paths:
                        for model_name, triton_path in triton_paths.items():
                            artifact_key = f"triton_{model_name}"
                            final_artifacts[artifact_key] = triton_path
                            logger.info(f"📦 Added Triton model artifact: {artifact_key} -> {triton_path}")
                        
                        logger.info(f"✅ Created {len(triton_paths)} Triton model directories!")
                else:
                    logger.info("  No Triton structure requested, using model directories as-is")
                
            except Exception as e:
                logger.error(f"❌ Failed to create Triton directories: {e}")
                # Note: Model directory artifacts are already added above as fallback
    
    try:
        # MLflow logging with model directory artifacts included
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
        
        if models_to_convert_onnx and model_result:
            if isinstance(model_result, dict):
                logger.info(f"✅ Model logged with {len(model_result)} model directories created!")
            else:
                logger.info("✅ Model logged with model directory created!")
        
        if not models_to_convert_onnx:
            logger.info("✅ Model logged successfully!")
            
    except Exception as e:
        logger.error(f"Error logging model: {e}")
        raise
    finally:
        # Note: Since we now create persistent model directories (not temporary files),
        # we don't need cleanup. The model directories are intentionally kept as artifacts.
        logger.debug("Model directories are preserved as artifacts (no cleanup needed)")


def create_model_directories_standalone(model_sources: Union[str, Dict[str, str], List[str]],
                                     output_dir: Optional[str] = None,
                                     create_triton_structure: bool = True) -> Dict[str, str]:
    """
    Create individual model directories from existing model files or directories.
    
    This is a standalone function that can be used independently of MLflow logging.
    
    Args:
        model_sources: Single model path, dict of {model_name: model_path}, or list of model paths
                      (can be files or directories)
        output_dir: Optional output directory (if None, creates directories in current working directory)
        create_triton_structure: If True, creates Triton-style directories. If False, uses source as-is.
        
    Returns:
        Dictionary mapping model names to their directory paths
        
    Example:
        # Single model directory with Triton structure
        model_paths = create_model_directories_standalone("my_model_dir", create_triton_structure=True)
        
        # Multiple model directories without Triton structure
        models = {"densenet_onnx": "encoder_dir", "bert_pytorch": "decoder_dir"}
        model_paths = create_model_directories_standalone(models, create_triton_structure=False)
        
        # Result structure (create_triton_structure=True):
        # densenet_onnx_triton/
        # └── 1/
        #     └── model.onnx
        #     └── [all other files from source directory]
        # bert_pytorch_triton/
        # └── 1/
        #     └── model.onnx 
        #     └── [all other files from source directory]
        
        # Result structure (create_triton_structure=False):
        # densenet_onnx/ (uses source directory as-is)
        # └── model.onnx
        # └── [all other files from source directory]
        # bert_pytorch/ (uses source directory as-is)
        # └── model.onnx
        # └── [all other files from source directory]
    """
    # Convert different input formats to consistent dict format
    if isinstance(model_sources, str):
        # Single model path (file or directory)
        model_name = Path(model_sources).stem
        models_dict = {model_name: model_sources}
    elif isinstance(model_sources, list):
        # List of model paths
        models_dict = {}
        for path in model_sources:
            model_name = Path(path).stem
            models_dict[model_name] = path
    elif isinstance(model_sources, dict):
        # Already in correct format
        models_dict = model_sources
    else:
        raise ValueError(f"Unsupported model_sources type: {type(model_sources)}")
    
    if create_triton_structure:
        logger.info(f"  Creating Triton model directories for {len(models_dict)} models")
    else:
        logger.info(f"  Using model directories for {len(models_dict)} models")
    
    # Change working directory if output_dir is specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    
    try:
        return _create_model_directories(
            model_dirs=models_dict,
            create_triton_structure=create_triton_structure
        )
    finally:
        # Restore original working directory
        if original_cwd:
            os.chdir(original_cwd)


