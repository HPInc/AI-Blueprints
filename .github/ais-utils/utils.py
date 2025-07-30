"""
AI-Blueprints MLflow Utilities with ONNX Integration and Model Directory Support

Provides enhanced MLflow logging capabilities with automatic ONNX export support
and individual model directory generation for ML pipelines.
Focused on Keras/TensorFlow and NeMo models for audio translation pipelines.

Uses ModelExportConfig for clean, per-model configuration.

Supported Model Types:
- 🎙️ NeMo: .nemo (ASR, TTS, Translation, etc.)
- 🧠 TensorFlow/Keras: .keras, .h5, .pb, SavedModel directories
- 🤖 PyTorch: .pt, .pth files
- 🔤 Transformers: Hugging Face models
- 📊 Scikit-learn: .pkl, .joblib files

Generated Artifacts:
- Original models (.nemo, .keras, etc.)
- ONNX versions (.onnx) with naming: {triton_model_name}.onnx
- External weights (.onnx.data) for models >2GB
- Individual model directories ready for deployment
- MLflow artifacts and signatures

Key Features:
- ✅ Per-model configuration with ModelExportConfig
- ✅ Automatic ONNX conversion from multiple model formats
- ✅ Individual model directories for each model
- ✅ Support for both single and multiple model pipelines
- ✅ Ready for deployment and easy model management
- ✅ Type safety and documentation
- ✅ Extensible for new model types

API Examples:

    # Individual model configuration
    from ais_utils.utils import ModelExportConfig, log_model
    
    model_configs = [
        ModelExportConfig(
            model_path="encoder.keras",
            model_name="bert_encoder",         # Model name for directory
            input_shape=(1, 224, 224, 3)
        ),
        ModelExportConfig(
            model_path="decoder.nemo",
            model_name="bert_decoder",         # Model name for directory
            input_sample=sample_audio
        ),
    ]
    
    log_model(
        artifact_path="bert_pipeline",
        python_model=BERTTourismModel(),
        artifacts={"corpus": "corpus.csv", "embeddings": "embeddings.csv"},
        models_to_convert_onnx=model_configs        # ONNX conversion + model directories
    )
    
    # Helper function for easier configuration
    from ais_utils.utils import create_model_export_configs
    
    configs = create_model_export_configs(
        models_dict={"bert_encoder": "encoder.keras", "bert_decoder": "decoder.nemo"},
        input_shapes={"bert_encoder": (1, 224, 224, 3)},
        input_samples={"bert_decoder": sample_audio}
    )

Generated Model Structure:
    MLflow Artifacts:
    ├── model_bert_encoder/                 # First model directory
    │   └── 1/
    │       └── model.onnx                 # ONNX model
    ├── model_bert_decoder/                # Second model directory  
    │   └── 1/
    │       └── model.onnx                 # ONNX model
    ├── onnx_bert_encoder.onnx             # Original ONNX files
    └── onnx_bert_decoder.onnx

Benefits:
- ✅ Individual model directories as MLflow artifacts
- ✅ Per-model input shapes and samples
- ✅ Better validation and error messages
- ✅ Cleaner, more maintainable code
- ✅ Type safety and documentation
- ✅ Extensible for new model types
- ✅ Easy model deployment and management"""

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
logger = logging.getLogger(__name__)


@dataclass
class ModelExportConfig:
    model_path: str
    model_name: str  # Name for the model directory
    input_shape: Optional[Tuple] = None
    input_sample: Optional[Any] = None
    model_type: Optional[str] = None  # Auto-detect if None
    task: str = "text-classification"  # for Transformers models
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-detect if type is None"""
        if self.model_type is None:
            self.model_type = _detect_model_type(self.model_path)
    
    def get_onnx_filename(self) -> str:
        """return ONNX filename."""
        return f"{self.model_name}.onnx"
    
    def validate(self) -> None:
        """Validade configuration."""
        if self.model_type in ['nemo', 'pytorch', 'sklearn'] and self.input_sample is None:
            raise ValueError(f"Input sample required for {self.model_type} model '{self.model_name}'.")
        
        if self.model_type == 'tensorflow' and self.input_shape is None:
            raise ValueError(f"Input shape required for TensorFlow model '{self.model_name}'.")

def create_model_export_configs(
    models_dict: Dict[str, str],
    input_shapes: Optional[Dict[str, Tuple]] = None,
    input_samples: Optional[Dict[str, Any]] = None
) -> List[ModelExportConfig]:

    configs = []
    
    for model_name, model_path in models_dict.items():
        config = ModelExportConfig(
            model_path=model_path,
            model_name=model_name,
            input_shape=input_shapes.get(model_name) if input_shapes else None,
            input_sample=input_samples.get(model_name) if input_samples else None
        )
        configs.append(config)
    
    return configs


def _detect_model_type(model_path: str) -> str:
    """
    Detect model type based on file extension.
    
    Returns:
        Model type: 'nemo', 'tensorflow', 'pytorch', 'sklearn', 'transformers', or 'unsupported'
    """
    path = Path(model_path)
    
    if path.suffix == '.nemo':
        return 'nemo'
    elif path.suffix in ['.keras', '.h5'] or path.name.endswith('.pb') or path.is_dir():
        # Check if it's a SavedModel directory
        if path.is_dir() and (path / 'saved_model.pb').exists():
            return 'tensorflow'
        elif path.suffix in ['.keras', '.h5'] or path.name.endswith('.pb'):
            return 'tensorflow'
        # Could be a transformers model directory
        elif path.is_dir() and (path / 'config.json').exists():
            return 'transformers'
        else:
            return 'tensorflow'  # Default assumption for directories
    elif path.suffix in ['.pt', '.pth']:
        return 'pytorch'
    elif path.suffix in ['.pkl', '.pickle', '.joblib']:
        return 'sklearn'
    elif path.name in ['pytorch_model.bin', 'model.safetensors']:
        return 'transformers'
    else:
        # Check if it's a model identifier (like "bert-base-uncased")
        if '/' in str(path) or '-' in str(path) and not path.exists():
            return 'transformers'
        else:
            return 'unsupported'


def _convert_single_model_to_onnx(config: ModelExportConfig) -> str:
    """
    Convert a single model to ONNX format using ModelExportConfig.
    
    Args:
        config: ModelExportConfig containing all model configuration
        
    Returns:
        Path to converted ONNX model
    """
    from onnx_export import (
        export_tensorflow_to_onnx, 
        export_nemo_to_onnx,
        export_pytorch_to_onnx,
        export_transformers_to_onnx,
        export_sklearn_to_onnx
    )
    
    # Validar configuração
    config.validate()
    
    onnx_filename = config.get_onnx_filename()
    logger.info(f"🔄 Converting {config.model_type} model: {config.model_path} -> {onnx_filename}")
    
    if config.model_type == 'tensorflow':
        return export_tensorflow_to_onnx(
            model_path=config.model_path,
            input_shape=config.input_shape,
            output_path=onnx_filename,
            model_name=config.model_name
        )
    elif config.model_type == 'nemo':
        return export_nemo_to_onnx(
            model_path=config.model_path,
            input_sample=config.input_sample,
            output_path=onnx_filename,
            model_name=config.model_name
        )
    elif config.model_type == 'pytorch':
        return export_pytorch_to_onnx(
            model_path=config.model_path,
            input_sample=config.input_sample,
            output_path=onnx_filename,
            model_name=config.model_name
        )
    elif config.model_type == 'transformers':
        return export_transformers_to_onnx(
            model_name_or_path=config.model_path,
            task=config.task,
            output_path=onnx_filename,
            model_name=config.model_name
        )
    elif config.model_type == 'sklearn':
        return export_sklearn_to_onnx(
            model_path=config.model_path,
            input_sample=config.input_sample,
            output_path=onnx_filename,
            model_name=config.model_name
        )
    else:
        raise ValueError(f"Unsupported model type '{config.model_type}' for model: {config.model_path}")


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





def _create_model_directories(onnx_models: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Create individual model directories with ONNX models for MLflow artifacts.
    
    Args:
        onnx_models: Single ONNX path or dict of {model_name: onnx_path}
        
    Returns:
        Dictionary mapping model_names to their directory paths
        
    Generated Structure:
        {model_name}/
        └── 1/
            └── model.onnx
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
    
    logger.info(f"� Creating model directories for {len(models_to_process)} models: {list(models_to_process.keys())}")
    
    for model_name, onnx_path in models_to_process.items():
        if not os.path.exists(onnx_path):
            logger.warning(f"⚠️ Model file not found: {onnx_path}")
            continue
        
        try:
            # Create model directory structure
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
            logger.info(f"📁 Created model directory: {model_name}/1/{Path(model_dest).name}")
            
            # Copy external data file if it exists (for ONNX models >2GB)
            onnx_data_path = f"{onnx_path}.data"
            if os.path.exists(onnx_data_path):
                data_dest = os.path.join(version_dir, f"{Path(model_dest).name}.data")
                shutil.copy2(onnx_data_path, data_dest)
                logger.info(f"📁 Added model data: {model_name}/1/{Path(model_dest).name}.data")
            
            model_paths[model_name] = model_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to create model directory for {model_name}: {e}")
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
            logger.info("� Creating individual model directories for artifacts...")
            
            try:
                model_paths = _create_model_directories(onnx_models=onnx_result)
                
                # Add each model directory as a separate artifact
                if model_paths:
                    for model_name, model_dir in model_paths.items():
                        artifact_key = f"model_{model_name}"
                        final_artifacts[artifact_key] = model_dir
                        logger.info(f"📦 Added model artifact: {artifact_key} -> {model_dir}")
                    
                    logger.info(f"✅ Created {len(model_paths)} model directories!")
                
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
                                     output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Create individual model directories from existing ONNX models.
    
    This is a standalone function that can be used independently of MLflow logging.
    
    Args:
        onnx_models: Single ONNX path, dict of {model_name: onnx_path}, or list of ONNX paths
        output_dir: Optional output directory (if None, creates directories in current working directory)
        
    Returns:
        Dictionary mapping model names to their directory paths
        
    Example:
        # Single model
        model_paths = create_model_directories_standalone("my_model.onnx")
        
        # Multiple models
        models = {"densenet_onnx": "encoder.onnx", "bert_pytorch": "decoder.pt"}
        model_paths = create_model_directories_standalone(models)
        
        # Result structure:
        # densenet_onnx/
        # └── 1/
        #     └── model.onnx
        # bert_pytorch/
        # └── 1/
        #     └── model.pt
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
    
    logger.info(f"� Creating model directories for {len(models_dict)} models")
    
    # Change working directory if output_dir is specified
    original_cwd = None
    if output_dir:
        original_cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    
    try:
        return _create_model_directories(onnx_models=models_dict)
    finally:
        # Restore original working directory
        if original_cwd:
            os.chdir(original_cwd)

def load_config_and_secrets(
    config_path: str = "../../configs/config.yaml",
    secrets_path: str = "../../configs/secrets.yaml"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration and secrets from YAML files.

    Args:
        config_path: Path to the configuration YAML file.
        secrets_path: Path to the secrets YAML file.

    Returns:
        Tuple containing (config, secrets) as dictionaries.

    Raises:
        FileNotFoundError: If either the config or secrets file is not found.
    """
    # Convert to absolute paths if needed
    config_path = os.path.abspath(config_path)
    secrets_path = os.path.abspath(secrets_path)

    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"secrets.yaml file not found in path: {secrets_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    with open(secrets_path) as file:
        secrets = yaml.safe_load(file)

    return config, secrets

def load_config(
    config_path: str = "../../configs/config.yaml"
) -> Dict[str, Any]:
    """
    Load configuration and secrets from YAML files.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Tuple containing (config) as dictionaries.

    Raises:
        FileNotFoundError: If either the config or secrets file is not found.
    """
    # Convert to absolute paths if needed
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml file not found in path: {config_path}")

    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config