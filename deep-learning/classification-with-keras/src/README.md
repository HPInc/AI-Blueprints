

# ONNX Export with MLflow - ModelExportConfig Quick Guide

This guide shows how to use `ModelExportConfig` for NeMo, PyTorch, Keras, and Transformers (translation) models. It also explains the required parameters for each case.

## How to Use

You must pass a list of `ModelExportConfig` objects to the `log_model` function from `onnx_utils` to enable ONNX export:

```python
from onnx_utils import ModelExportConfig, log_model

model_configs = [
    ModelExportConfig(...),  # see below for each framework
    # You can add more models here if needed
]

log_model(
    artifact_path="my_model",
    python_model=MyPythonModel(),
    artifacts=artifacts_dict,
    signature=model_signature,
    models_to_convert_onnx=model_configs  # List of ModelExportConfig
)
```

---

## ModelExportConfig Examples by Framework

### 1. NeMo - Audio Translation with Nemo

```python
mt_model = MarianMTModel.from_pretrained(MT_MODEL)
asr_model = nemo_asr.models.EncDecCTCModel.restore_from(nemo_models["enc_dec_CTC"])
fast_pitch_model = nemo_tts.models.FastPitchModel.restore_from(nemo_models["fast_pitch"])
hifi_gan_model = nemo_tts.models.HifiGanModel.restore_from(nemo_models["hifi_gan"])
      
model_configs = [ 
            ModelExportConfig(
                model=mt_model,                         # 🚀 Pre-loaded Transformers model!
                model_name="Helsinki-NLP",              # ONNX file naming
                task="translation",                     # Model task
            ),
            # NeMo ASR model
            ModelExportConfig(
                model=asr_model.to(device),                        # 🚀 Pre-loaded NeMo ASR model!
                model_name="enc_dec_CTC",               # ONNX file naming
            ),
            # NeMo FastPitch model
            ModelExportConfig(
                model=fast_pitch_model.to(device),                 # 🚀 Pre-loaded NeMo TTS model!
                model_name="fast_pitch",                # ONNX file naming
            ),
            # NeMo HifiGAN model
            ModelExportConfig(
                model=hifi_gan_model.to(device),                   # 🚀 Pre-loaded NeMo Vocoder model!
                model_name="hifi_gan",                  # ONNX file naming
            ),
        ] 
```

If your NeMo model does not support export, you should try to wrap it as a PyTorch model and try to use the torch converter, config example:

---

### 2. PyTorch (using wrapped BERT)

```python
bert_model = ...  # Your loaded PyTorch BERT model
input_ids = torch.randint(0, 30522, (1, 128))
attention_mask = torch.ones((1, 128))
token_type_ids = torch.zeros((1, 128))
config = ModelExportConfig(
    model=bert_model,
    model_name="bert_pytorch",
    input_sample=(input_ids, attention_mask, token_type_ids),
    opset=14,           
    do_constant_folding=True,   # PyTorch-specific
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    },
    verbose=True
)
```

---

### 3. Keras (TensorFlow)

```python
import tensorflow as tf
keras_model = tf.keras.models.load_model("model.keras")
input_sample = tf.random.normal((1, 28, 28, 1))
config = ModelExportConfig(
    model=keras_model,
    model_name="keras_classifier",
    input_sample=input_sample,
    opset=12,                
    use_saved_model=False,   # Set True for complex/custom models or Keras 3
    verbose=False
)
```

---

### 4. Transformers (Translation)

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-pt")
config = ModelExportConfig(
    model=model,
    model_name="opus_translator",
    input_sample=None,         # Not required for Transformers
    task="translation",       # Required for translation
    opset=14,                 
    feature="seq2seq-lm",     # Transformers-specific
    verbose=False
)
```

---

## Parameter Reference

Below are the most relevant parameters for each framework:

| Framework      | Parameters                                                                 |
|--------------- |----------------------------------------------------------------------------|
| **NeMo**       | `export_format`, `check_trace`, `verbose`, `input_sample`                  |
| **PyTorch**    | `opset`, `do_constant_folding`, `input_names`, `output_names`, `dynamic_axes`, `verbose`, `input_sample` |
| **Keras**      | `opset`, `use_saved_model`, `verbose`, `input_sample`                      |
| **Transformers (translation)** | `task`, `opset`, `feature`, `verbose`                      |

**Notes:**
- For Keras 3 or custom models, set `use_saved_model=True`.
- For Transformers translation, always set `task="translation"` and `feature="seq2seq-lm"`.
---
