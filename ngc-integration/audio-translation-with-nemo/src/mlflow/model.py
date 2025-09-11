"""
Standalone Model class for NeMo Audio Translation.

Business Logic Layer
- Handles end-to-end audio translation using NVIDIA NeMo models
- Manages ASR, machine translation, and TTS model initialization and prediction logic  
- Contains all domain-specific functionality without MLflow dependencies
- Designed to be framework-agnostic and easily testable
"""

import os
import io
import uuid
import base64
import logging
from typing import Dict, Any, Optional, List
import shutil
import numpy as np
import pandas as pd
import soundfile
import torch
import warnings

# NeMo imports
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts

# Transformers imports  
from transformers import MarianMTModel, MarianTokenizer
import logging
logging.getLogger("nemo").setLevel(logging.ERROR)


# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles end-to-end audio translation using NVIDIA NeMo models.
    """

    def __init__(self, config: dict, docs_path: str, secrets: dict = None, model_path: str = None):
        """
        Initialize the Model with vanilla-rag compatible interface.
        Internally maps to NeMo-specific requirements.

        Args:
            config: Model configuration dictionary (contains nemo_models paths)
            docs_path: Path to documents directory (unused for NeMo but required for interface)
            secrets: Secrets dictionary (unused for NeMo but required for interface)
            model_path: Single model path (unused for NeMo but required for interface)
        """
        self.model_config = config
        self.docs_path = docs_path
        self.secrets = secrets
        self.model_path = model_path

        # Extract NeMo-specific configuration from config
        # In artifact context, NeMo models are stored directly in data_path
        # In development context, they're in config["nemo_models"]
        self.nemo_models = self._resolve_nemo_models()
        self.model_dir = os.path.dirname(docs_path) if docs_path else ""

        # Model components
        self.asr_model = None
        self.mt_tokenizer = None
        self.mt_model = None
        self.spectrogram_generator = None
        self.vocoder = None
        
        # Configuration
        self.mt_model_name = "Helsinki-NLP/opus-mt-en-es"
        self.framerate = 41000

        # Setup environment and load components
        try:
            self._setup_environment()
            self._load_models()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Model: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def _resolve_nemo_models(self) -> dict:
        """
        Resolve NeMo model paths from either artifacts or configuration.
        
        Returns:
            Dictionary mapping NeMo model names to their file paths
        """
        # First, check if we're in artifact context 
        # The vanilla-rag loader places model_path contents in data_path/models/
        if self.docs_path:
            artifact_dir = os.path.dirname(self.docs_path)  # This is the data_path
            
            # Try models subdirectory first (where vanilla-rag puts model_path contents)
            models_subdir = os.path.join(artifact_dir, "models")
            if os.path.exists(models_subdir):
                models_artifact_models = {
                    "enc_dec_CTC": os.path.join(models_subdir, "enc_dec_CTC.nemo"),
                    "fast_pitch": os.path.join(models_subdir, "fast_pitch.nemo"),
                    "hifi_gan": os.path.join(models_subdir, "hifi_gan.nemo")
                }
                
                if all(os.path.exists(path) for path in models_artifact_models.values()):
                    logger.info("Using NeMo models from MLflow artifacts/models subdirectory")
                    return models_artifact_models
            
            # Fallback: try direct in artifact_dir (data_path root)
            direct_artifact_models = {
                "enc_dec_CTC": os.path.join(artifact_dir, "enc_dec_CTC.nemo"),
                "fast_pitch": os.path.join(artifact_dir, "fast_pitch.nemo"),
                "hifi_gan": os.path.join(artifact_dir, "hifi_gan.nemo")
            }
            
            if all(os.path.exists(path) for path in direct_artifact_models.values()):
                logger.info("Using NeMo models from MLflow artifacts root")
                return direct_artifact_models
        
        # Fallback to config nemo_models
        config_nemo_models = self.config.get("nemo_models")
        if config_nemo_models:
            logger.info("Using NeMo model paths from configuration")
            return config_nemo_models
            
        # Last fallback - raise error
        raise ValueError("No NeMo model paths found in artifacts or configuration")

    def _setup_environment(self) -> None:
        """Configure environment variables and suppress verbose logs."""
        try:
            # Load secrets into environment if provided
            if self.secrets:
                for key, value in self.secrets.items():
                    os.environ[key] = str(value)
                logger.info("Secrets loaded into environment")
            
            # Configure proxy if specified in config
            if "proxy" in self.model_config and self.model_config["proxy"]:
                logger.info(f"Setting up proxy: {self.model_config['proxy']}")
                os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
                os.environ["HTTP_PROXY"] = self.model_config["proxy"]
            else:
                logger.info("No proxy configuration found")
                    
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function

    def _load_models(self) -> None:
        """Load all NeMo and Transformers models."""
        try:
            # Validate that all required NeMo model files exist
            for model_name, model_path in self.nemo_models.items():
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Required NeMo model file not found: {model_path}")
                logger.info(f"Found {model_name} model at: {model_path}")

            # Load NeMo models using resolved paths
            self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(self.nemo_models["enc_dec_CTC"])
            self.spectrogram_generator = nemo_tts.models.FastPitchModel.restore_from(self.nemo_models["fast_pitch"])
            self.vocoder = nemo_tts.models.HifiGanModel.restore_from(self.nemo_models["hifi_gan"])
            
            # Load Transformers models
            self.mt_tokenizer = MarianTokenizer.from_pretrained(self.mt_model_name)
            self.mt_model = MarianMTModel.from_pretrained(self.mt_model_name)

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def transcribe_audio(self, model_input: Dict[str, Any]) -> str:
        """
        Deserialize base64-encoded audio, save it temporarily, and perform speech-to-text.
        
        Args:
            model_input: Input dictionary containing serialized audio data
            
        Returns:
            Transcribed text string
        """
        try:
            serialized_audio = model_input['source_serialized_audio'][0]
            audio_buffer = io.BytesIO(base64.b64decode(serialized_audio))
            audio_array, self.framerate = soundfile.read(audio_buffer)

            # Ensure mono-channel audio
            if audio_array.ndim > 1:
                audio_array = audio_array[:, 0]

            # Generate unique file ID for this request
            file_id = uuid.uuid1()
            temp_wave_path = f"/phoenix/mlflow/tmp/{file_id}.wav"
            soundfile.write(temp_wave_path, audio_array, self.framerate)

            # Perform ASR
            transcribed_text = self.asr_model.cuda().transcribe([temp_wave_path])
            
            # Clean up temporary file
            if os.path.exists(temp_wave_path):
                os.remove(temp_wave_path)
                
            return transcribed_text[0] if transcribed_text else ""
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    def translate_text(self, source_text: str) -> str:
        """
        Translate text using Hugging Face MarianMT model.
        
        Args:
            source_text: Text to translate
            
        Returns:
            Translated text string
        """
        try:
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mt_model = self.mt_model.to(device)

            # Tokenize and move inputs to device
            inputs = self.mt_tokenizer(source_text, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate translation
            translated = self.mt_model.generate(**inputs)
            translated_text = self.mt_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return translated_text
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise

    def text_to_audio(self, text: str) -> np.ndarray:
        """
        Generate audio waveform from text using TTS models.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            parsed_tokens = self.spectrogram_generator.cuda().parse(text)
            spectrogram = self.spectrogram_generator.cuda().generate_spectrogram(tokens=parsed_tokens, speaker=2)
            audio_tensor = self.vocoder.cuda().convert_spectrogram_to_audio(spec=spectrogram)

            return audio_tensor.to('cpu').detach().numpy()
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise

    def serialize_audio(self, audio_array: np.ndarray, file_id: str) -> str:
        """
        Serialize a NumPy audio array into a base64-encoded WAV file.
        
        Args:
            audio_array: Audio data as numpy array
            file_id: Unique identifier for temporary files
            
        Returns:
            Base64-encoded audio string
        """
        try:
            # Save temporary file for reference (optional)
            wave_path = f"/phoenix/mlflow/tmp/out_{file_id}.wav"
            soundfile.write(wave_path, audio_array, samplerate=self.framerate, format='WAV')

            # Create base64 encoded version
            with io.BytesIO() as buffer:
                soundfile.write(buffer, audio_array, samplerate=self.framerate, format='WAV')
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            return audio_base64
        except Exception as e:
            logger.error(f"Error serializing audio: {str(e)}")
            raise

    def predict(self, model_input, params=None):
        """
        Process inputs and generate responses.
        Performs end-to-end audio translation pipeline.

        Args:
            model_input: Input data containing source text or audio
            params: Optional parameters (use_audio flag)

        Returns:
            pandas.DataFrame with translation results
        """
        try:
            # Initialize default parameters
            if params is None:
                params = {}

            # Generate unique file ID for this request
            file_id = str(uuid.uuid1())
            use_audio = params.get("use_audio", False)

            # Step 1: Get source text (either from audio transcription or direct input)
            if use_audio:
                source_text = self.transcribe_audio(model_input)
                logger.info(f"Transcribed audio to text: {source_text}")
            else:
                source_text = model_input['source_text'][0]
                logger.info(f"Using direct text input: {source_text}")

            # Step 2: Translate text
            translated_text = self.translate_text(source_text)
            logger.info(f"Translated text: {translated_text}")

            # Step 3: Generate audio from translated text (if requested)
            translated_audio_base64 = ""
            if use_audio:
                audio_array = self.text_to_audio(translated_text)
                translated_audio_base64 = self.serialize_audio(audio_array[0], file_id)
                logger.info("Generated audio from translated text")

            # Return results in expected format
            result = {
                "original_text": source_text,
                "translated_text": translated_text,
                "translated_serialized_audio": translated_audio_base64,
            }

            return pd.DataFrame([result])

        except Exception as e:
            import traceback
            logger.error(f"Error in predict: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error result in expected format
            error_result = {
                "original_text": "Error",
                "translated_text": f"Translation error: {str(e)}",
                "translated_serialized_audio": "",
            }
            return pd.DataFrame([error_result])
        
    def get_onnx_export_config(self) -> List:
        """
        Get configuration for ONNX export.
        Returns the configuration needed for ONNX model export.
        
        Returns:
            List of ModelExportConfig objects for ONNX conversion
        """
        try:
            # Import here to avoid circular imports
            from ..onnx_utils import ModelExportConfig
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use already loaded models from the instance
            model_configs = [
                ModelExportConfig(
                    model=self.mt_model,                    # 🚀 Pre-loaded Transformers model!
                    model_name="Helsinki-NLP",              # ONNX file naming
                    task="translation",                     # Model task
                ),
                # NeMo ASR model
                ModelExportConfig(
                    model=self.asr_model.to(device),        # 🚀 Pre-loaded NeMo ASR model!
                    model_name="enc_dec_CTC",               # ONNX file naming
                ),
                # NeMo FastPitch model
                ModelExportConfig(
                    model=self.spectrogram_generator.to(device),  # 🚀 Pre-loaded NeMo TTS model!
                    model_name="fast_pitch",                # ONNX file naming
                ),
                # NeMo HifiGAN model
                ModelExportConfig(
                    model=self.vocoder.to(device),          # 🚀 Pre-loaded NeMo Vocoder model!
                    model_name="hifi_gan",                  # ONNX file naming
                ),
            ]
            
            logger.info("ONNX export configuration created successfully")
            return model_configs
            
        except Exception as e:
            logger.error(f"Error creating ONNX export configuration: {str(e)}")
            raise RuntimeError(f"Failed to create ONNX export configuration: {str(e)}") from e

    def copy_nemo_models_to_directory(self, target_dir: str) -> None:
        """
        Copy NeMo model artifacts to a target directory.
        
        Args:
            target_dir: Directory path where to copy the NeMo model files
        """
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy NeMo model artifacts using the resolved paths
            if "enc_dec_CTC" in self.nemo_models:
                source_path = self.nemo_models["enc_dec_CTC"]
                target_path = os.path.join(target_dir, "enc_dec_CTC.nemo")
                if os.path.exists(source_path):
                    shutil.copyfile(source_path, target_path)
                    logger.info(f"Copied ASR model to {target_path}")
                    
            if "fast_pitch" in self.nemo_models:
                source_path = self.nemo_models["fast_pitch"]
                target_path = os.path.join(target_dir, "fast_pitch.nemo")
                if os.path.exists(source_path):
                    shutil.copyfile(source_path, target_path)
                    logger.info(f"Copied FastPitch model to {target_path}")
                    
            if "hifi_gan" in self.nemo_models:
                source_path = self.nemo_models["hifi_gan"]
                target_path = os.path.join(target_dir, "hifi_gan.nemo")
                if os.path.exists(source_path):
                    shutil.copyfile(source_path, target_path)
                    logger.info(f"Copied HifiGAN model to {target_path}")
                    
            logger.info(f"NeMo models copied to directory: {target_dir}")
            
        except Exception as e:
            logger.error(f"Error copying NeMo models: {str(e)}")
            raise RuntimeError(f"Failed to copy NeMo models: {str(e)}") from e