# Import NeMo and it's ASR, NLP and TTS collections
#import nemo
# Import Speech Recognition collection
#import nemo.collections.asr as nemo_asr
# Import Natural Language Processing collection
#import nemo.collections.nlp as nemo_nlp
# Import Speech Synthesis collection
#import nemo.collections.tts as nemo_tts
# We'll use this to listen to audio
#import IPython

import soundfile
import uuid
import io
import base64
import json
import logging
import shutil

import logging
import pip

import mlflow
import os
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature


class NemoTranslationModel(mlflow.pyfunc.PythonModel):
    def transcribe_audio(self, inputs):
        """
        Deserializes base64-serialized audio to a NumPy array.
        Assume the audio is in WAV format for simplicity
        """
        serialized_audio = inputs['source_serialized_audio'][0]
        audio_buffer = io.BytesIO(base64.b64decode(serialized_audio))
        audio, self.framerate = soundfile.read(audio_buffer)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio[:, 0] #Get single channel  
        wave_file = "/phoenix/mlflow/tmp/{}.wav".format(self.file_id)
        soundfile.write(wave_file, audio, self.framerate)
        text = self.asr_model.cuda().transcribe([wave_file])
        return text

    def text_to_audio(self, text):
        """
        Generates audio from text using TTS templates.
        """
        parsed = self.spectrogram_generator.cuda().parse(text)
        spectrogram = self.spectrogram_generator.cuda().generate_spectrogram(tokens=parsed, speaker=2)
        audio = self.vocoder.cuda().convert_spectrogram_to_audio(spec=spectrogram)
        return audio.to('cpu').detach().numpy()

    def serialize_audio(self, audio_np):
        """
        Serializes an audio NumPy array to a base64 string representing a WAV file.
        """
        audio_base64 = ""
        wave_file = "/phoenix/mlflow/tmp/out_{}.wav".format(self.file_id)
        soundfile.write(wave_file, audio_np, samplerate=41000, format='WAV')
        
        with io.BytesIO() as audio_buffer:
            soundfile.write(audio_buffer, audio_np, samplerate=41000, format='WAV')
            audio_buffer.seek(0)
            audio_output = audio_buffer.read()
        return audio_output

    def load_dependencies(self):
        logging.info("----------------------------")
        logging.info("Starting to load the model context.")
        logging.info("----------------------------")
        pip.main(["install", 
                  "nemo-toolkit==1.23.0", 
                  "transformers==4.40.2", 
                  "huggingface-hub==0.23.2", 
                  "youtokentome", 
                  "pytorch-lightning==2.0.9", 
                  "megatron_core==0.5.0"])
        logging.info("Fixed dependencies.")
        logging.info("----------------------------")
        if os.path.isdir("/opt/megatron-lm"):
            shutil.rmtree("/opt/megatron-lm")
        logging.info("----------------------------")


    def load_context(self, context):
        self.load_dependencies()
        import nemo.collections.asr as nemo_asr
        import nemo.collections.nlp as nemo_nlp
        import nemo.collections.tts as nemo_tts
        model_name=context.artifacts["model"]
        self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from("{}/enc_dec_CTC.nemo".format(model_name))
        self.nmt_model = nemo_nlp.models.MTEncDecModel.restore_from("{}/MT_enc_dec.nemo".format(model_name))
        self.spectrogram_generator = nemo_tts.models.FastPitchModel.restore_from("{}/fast_pitch.nemo".format(model_name))
        self.vocoder = nemo_tts.models.HifiGanModel.restore_from("{}/hifi_gan.nemo".format(model_name))
        self.framerate = 41000
        if not os.path.isdir("/phoenix/mlflow/tmp"):
            os.mkdir("/phoenix/mlflow/tmp")
         
    def predict(self, context, model_input, params):
        source_text = ""
        self.file_id = uuid.uuid1()
        if params["use_audio"]:
            source_text = self.transcribe_audio(model_input)[0]
        else:
            source_text = model_input['source_text'][0]
        translated_text = self.nmt_model.cuda().translate([source_text])[0]
        translated_audio = ""
        if params["use_audio"]:
            audio = self.text_to_audio(translated_text)
            translated_audio = self.serialize_audio(audio[0])
        return {"original_text": source_text, "translated_text": translated_text, "translated_serialized_audio": translated_audio}

    @classmethod
    def log_model(cls, model_name, nemo_models, demo_folder): #eg (model, '', 'my_model')
        import os, shutil
        input_schema = Schema(
            [
                ColSpec("string", "source_text"),
                ColSpec("string", "source_serialized_audio"),
            ]
        )
        output_schema = Schema(
            [
                ColSpec("string", "original_text"),
                ColSpec("string", "translated_text"),
                ColSpec("string", "translated_serialized_audio"),
            ]
        )
        
        params_schema = ParamSchema(
            [
                ParamSpec("use_audio", "boolean", False)
            ]
        )
      
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

        if not os.path.isdir(model_name):
            os.mkdir(model_name)
            shutil.copyfile(nemo_models["enc_dec_CTC"], "{}/enc_dec_CTC.nemo".format(model_name))
            shutil.copyfile(nemo_models["MT_enc_dec"], "{}/MT_enc_dec.nemo".format(model_name))
            shutil.copyfile(nemo_models["fast_pitch"], "{}/fast_pitch.nemo".format(model_name))
            shutil.copyfile(nemo_models["hifi_gan"], "{}/hifi_gan.nemo".format(model_name))
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts={"model": model_name, "demo": demo_folder},
            signature=signature
        )            
        
        shutil.rmtree(model_name)


mlflow.set_experiment(experiment_name='NeMo_translation')

with mlflow.start_run(run_name='NeMo_en_es_translation') as run:
    model_set = {
        "enc_dec_CTC": "/home/jovyan/datafabric/STT_En_Citrinet_1024_Gamma_0_25/stt_en_citrinet_1024_gamma_0_25.nemo",
        "MT_enc_dec": "/home/jovyan/datafabric/NMT_En_Es_Transformer12x2/nmt_en_es_transformer12x2.nemo",
        "fast_pitch": "/home/jovyan/datafabric/TTS_Es_Multispeaker_FastPitch_HiFiGAN/tts_es_fastpitch_multispeaker.nemo",
        "hifi_gan": "/home/jovyan/datafabric/TTS_Es_Multispeaker_FastPitch_HiFiGAN/tts_es_hifigan_ft_fastpitch_multispeaker.nemo",
    }

    NemoTranslationModel.log_model(model_name='nemo_en_es', nemo_models=model_set, demo_folder = "demo")
    mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/nemo_en_es", name="nemo_en_es")

    
