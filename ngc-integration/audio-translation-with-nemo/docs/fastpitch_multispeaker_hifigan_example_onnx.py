import json
import requests
import numpy as np
import soundfile as sf 

# === Configurations ===
FASTPITCH_URL = "http://localhost:8000/v2/models/fast_pitch/infer"
HIFIGAN_URL = "http://localhost:8000/v2/models/hifi_gan/infer"
VOCAB_PATH = "fastpitch_vocab.json"
TEXT = "mi mamá siempre decía que la vida como una caja de chocolates nunca sabe lo que vas a conseguir"
SPEAKER_ID = 2  # or another value supported by the model

# === Load vocabulary ===
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

char2id = {c: i for i, c in enumerate(vocab)}
text_tokens = [char2id.get(c, char2id[" "]) for c in TEXT.lower()]
text_len = len(text_tokens)

pitch_array = np.random.normal(loc=1.0, scale=0.005, size=len(text_tokens)).tolist()

# === FASTPITCH inference ===
fastpitch_payload = {
    "inputs": [
        {
            "name": "input.1",
            "shape": [1],
            "datatype": "INT64",
            "data": [SPEAKER_ID]
        },
        {
            "name": "pace",
            "shape": [1, 1],
            "datatype": "FP32",
            "data": [1.0]
        },
       {
    "name": "pitch",
    "shape": [1, len(text_tokens)],
    "datatype": "FP32",
    "data": pitch_array
},
        {
            "name": "text",
            "shape": [1, text_len],
            "datatype": "INT64",
            "data": text_tokens
        }
    ],
    "outputs": [{"name": "spect"}]
}

fastpitch_response = requests.post(
    FASTPITCH_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(fastpitch_payload)
)
fastpitch_response.raise_for_status()
spect_output = next(o for o in fastpitch_response.json()["outputs"] if o["name"] == "spect")["data"]
print("=========================== Fastpich Response (===========================")
print(spect_output)
print("=========================== End Fastpich Response (===========================")
print("")

# === Prepare spectrogram for HiFi-GAN ===
mel_bins = 80
num_frames = len(spect_output) // mel_bins
spect_array = np.array(spect_output, dtype=np.float32).reshape((1, mel_bins, num_frames))  # shape [1, 80, N]

# === HIFIGAN inference ===
hifigan_payload = {
    "inputs": [
        {
            "name": "spec",
            "shape": list(spect_array.shape),
            "datatype": "FP32",
            "data": spect_array.flatten().tolist()
        }
    ],
    "outputs": [{"name": "audio"}]
}

hifigan_response = requests.post(
    HIFIGAN_URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(hifigan_payload)
)
hifigan_response.raise_for_status()

audio_data = next(o for o in hifigan_response.json()["outputs"] if o["name"] == "audio")["data"]
print("=========================== Hifigan Response (===========================")
print(audio_data)
print("=========================== End Response (===========================")

