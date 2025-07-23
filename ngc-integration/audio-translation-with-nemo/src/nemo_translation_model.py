"""
NeMo Translation Model for MLflow deployment.
This implementation uses code-based logging to avoid serialization issues.
"""

import os
import shutil
import uuid
import io
import base64
import warnings
import logging
import numpy as np
import torch
import soundfile

# NeMo Core Imports
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts

# Transformers
from transformers import MarianMTModel, MarianTokenizer

# MLflow Integration
import mlflow
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature


class NemoTranslationModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow pyfunc model for performing end-to-end audio translation using NVIDIA NeMo models.
    
    This implementation avoids serialization issues by using code-based logging.
    """

    def __init__(self):
        """Initialize the model without loading heavy objects."""
        self.asr_model = None
        self.mt_tokenizer = None
        self.mt_model = None
        self.spectrogram_generator = None
        self.vocoder = None
        self.framerate = 41000
        self.mt_model_name = "Helsinki-NLP/opus-mt-en-es"

    def load_context(self, context):
        """Load NeMo models and prepare the temporary working directory."""
        # Suppress verbose logs
        warnings.filterwarnings("ignore")
        logging.getLogger('nemo_logger').setLevel(logging.ERROR)
        
        model_dir = context.artifacts["model"]

        # Load models only when needed to avoid serialization issues
        self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(f"{model_dir}/enc_dec_CTC.nemo")
        self.mt_tokenizer = MarianTokenizer.from_pretrained(self.mt_model_name)
        self.mt_model = MarianMTModel.from_pretrained(self.mt_model_name)
        self.spectrogram_generator = nemo_tts.models.FastPitchModel.restore_from(f"{model_dir}/fast_pitch.nemo")
        self.vocoder = nemo_tts.models.HifiGanModel.restore_from(f"{model_dir}/hifi_gan.nemo")

        # Create temp directory - use /phoenix/mlflow/tmp if it exists, otherwise use /tmp
        temp_dir = "/phoenix/mlflow/tmp" if os.path.exists("/phoenix") else "/tmp/mlflow_nemo"
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    def transcribe_audio(self, model_input):
        """Deserialize base64-encoded audio, save it temporarily, and perform speech-to-text."""
        serialized_audio = model_input['source_serialized_audio'][0]
        audio_buffer = io.BytesIO(base64.b64decode(serialized_audio))
        audio_array, self.framerate = soundfile.read(audio_buffer)

        # Ensure mono-channel audio
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]

        temp_wave_path = f"{self.temp_dir}/{self.file_id}.wav"
        soundfile.write(temp_wave_path, audio_array, self.framerate)

        # Perform ASR
        transcribed_text = self.asr_model.cuda().transcribe([temp_wave_path])
        
        # Clean up temp file
        if os.path.exists(temp_wave_path):
            os.remove(temp_wave_path)
            
        return transcribed_text

    def text_to_audio(self, text: str):
        """Generate audio waveform from text using TTS models."""
        parsed_tokens = self.spectrogram_generator.cuda().parse(text)
        spectrogram = self.spectrogram_generator.cuda().generate_spectrogram(tokens=parsed_tokens, speaker=2)
        audio_tensor = self.vocoder.cuda().convert_spectrogram_to_audio(spec=spectrogram)

        return audio_tensor.to('cpu').detach().numpy()

    def serialize_audio(self, audio_array: np.ndarray):
        """Serialize a NumPy audio array into a base64-encoded WAV file."""
        with io.BytesIO() as buffer:
            soundfile.write(buffer, audio_array, samplerate=self.framerate, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return audio_base64

    def predict(self, context, model_input, params):
        """
        Perform inference:
        1. Transcribe audio (if input is audio)
        2. Translate text using Hugging Face MarianMT
        3. Synthesize translated text into speech
        4. Serialize the audio if needed
        """
        self.file_id = str(uuid.uuid1())
        use_audio = params.get("use_audio", False) if params else False

        try:
            if use_audio:
                source_text = self.transcribe_audio(model_input)[0]
            else:
                source_text = model_input['source_text'][0]

            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mt_model = self.mt_model.to(device)

            # Tokenize and move inputs to device
            inputs = self.mt_tokenizer(source_text, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate translation
            translated = self.mt_model.generate(**inputs)
            translated_text = self.mt_tokenizer.decode(translated[0], skip_special_tokens=True)

            translated_audio_base64 = ""
            if use_audio:
                audio_array = self.text_to_audio(translated_text)
                translated_audio_base64 = self.serialize_audio(audio_array[0])

            return {
                "original_text": source_text,
                "translated_text": translated_text,
                "translated_serialized_audio": translated_audio_base64
            }
        
        except Exception as e:
            return {
                "original_text": "",
                "translated_text": f"Error during translation: {str(e)}",
                "translated_serialized_audio": ""
            }

    @classmethod
    def log_model(cls, model_name: str, nemo_models: dict, demo_folder: str, pip_requirements=None):
        """
        Log the translation model to MLflow using code-based logging.
        
        Args:
            model_name: Name under which to register the model.
            nemo_models: Dictionary mapping component names to their local .nemo file paths.
            demo_folder: Path to the demo files folder.
            pip_requirements: Requirements for the model.
        """
        
        # Define model signature
        input_schema = Schema([
            ColSpec("string", "source_text"),
            ColSpec("string", "source_serialized_audio"),
        ])

        output_schema = Schema([
            ColSpec("string", "original_text"),
            ColSpec("string", "translated_text"),
            ColSpec("string", "translated_serialized_audio"),
        ])

        params_schema = ParamSchema([
            ParamSpec("use_audio", "boolean", False)
        ])

        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=params_schema
        )

        # Create model directory
        os.makedirs(model_name, exist_ok=True)

        # Copy NeMo model artifacts
        if "enc_dec_CTC" in nemo_models:
            shutil.copyfile(nemo_models["enc_dec_CTC"], f"{model_name}/enc_dec_CTC.nemo")
        if "fast_pitch" in nemo_models:
            shutil.copyfile(nemo_models["fast_pitch"], f"{model_name}/fast_pitch.nemo")
        if "hifi_gan" in nemo_models:
            shutil.copyfile(nemo_models["hifi_gan"], f"{model_name}/hifi_gan.nemo")

        # Use code-based logging instead of object serialization
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            code_paths=["../core", "../src"],
            python_model=cls(),
            artifacts={"model": model_name, "demo": demo_folder},
            signature=signature,
            pip_requirements=pip_requirements
        )

        # Clean up temporary files
        shutil.rmtree(model_name)


def _load_pyfunc(path):
    """
    Load the model for MLflow serving.
    This function is required for code-based logging.
    """
    return NemoTranslationModel()
