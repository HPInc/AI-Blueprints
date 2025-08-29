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
from typing import Dict, Any, Optional
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

# Set up logger
logger = logging.getLogger(__name__)


class Model:
    """
    Standalone model class with no MLflow inheritance.
    Handles end-to-end audio translation using NVIDIA NeMo models.
    """

    def __init__(self, config: dict, nemo_models: dict, model_dir: str = None):
        """
        Initialize the Model with configuration and model artifacts.

        Args:
            config: Model configuration dictionary
            nemo_models: Dictionary mapping component names to their model file paths
            model_dir: Path to model artifacts directory (for MLflow context)
        """
        self.config = config
        self.nemo_models = nemo_models
        self.model_dir = model_dir if model_dir else ""

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

    def _setup_environment(self) -> None:
        """Configure environment variables and suppress verbose logs."""
        try:
            # Suppress warnings and verbose logs
            warnings.filterwarnings("ignore")
            logging.getLogger('nemo_logger').setLevel(logging.ERROR)
            
            # Create temporary directory for processing
            os.makedirs("/phoenix/mlflow/tmp", exist_ok=True)
            
            logger.info("Environment setup completed")
        except Exception as e:
            logger.error(f"Error setting up environment: {str(e)}")
            # Continue without failing to allow the model to still function

    def _load_models(self) -> None:
        """Load all NeMo and Transformers models."""
        try:
            # Load ASR model
            if self.model_dir:
                # MLflow context - load from artifacts
                asr_path = f"{self.model_dir}/enc_dec_CTC.nemo"
                spectrogram_path = f"{self.model_dir}/fast_pitch.nemo" 
                vocoder_path = f"{self.model_dir}/hifi_gan.nemo"
            else:
                # Direct initialization - use provided paths
                asr_path = self.nemo_models.get("enc_dec_CTC", "")
                spectrogram_path = self.nemo_models.get("fast_pitch", "")
                vocoder_path = self.nemo_models.get("hifi_gan", "")

            # Load NeMo models
            self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(asr_path)
            self.spectrogram_generator = nemo_tts.models.FastPitchModel.restore_from(spectrogram_path)
            self.vocoder = nemo_tts.models.HifiGanModel.restore_from(vocoder_path)
            
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