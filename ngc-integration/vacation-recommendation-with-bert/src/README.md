

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

### 1. NeMo

```python
from nemo.collections.nlp.models import BERTLMModel
nemo_model = BERTLMModel.restore_from("bert_model.nemo")
input_sample = {
    'input_ids': torch.randint(0, 30522, (1, 128)),
    'attention_mask': torch.ones((1, 128))
}
config = ModelExportConfig(
    model=nemo_model,
    model_name="nemo_bert_model",
    input_sample=input_sample,  
    check_trace=True,           # NeMo-specific
    verbose=False
)
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
    opset_version=14,           # PyTorch-specific
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
| **PyTorch**    | `opset_version`, `do_constant_folding`, `input_names`, `output_names`, `dynamic_axes`, `verbose`, `input_sample` |
| **Keras**      | `opset`, `use_saved_model`, `verbose`, `input_sample`                      |
| **Transformers (translation)** | `task`, `opset`, `feature`, `verbose`                      |

**Notes:**
- For Keras 3 or custom models, set `use_saved_model=True`.
- For Transformers translation, always set `task="translation"` and `feature="seq2seq-lm"`.

---
