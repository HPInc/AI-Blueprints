import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import setup_chat_format

def merge_lora_and_save(
    base_model_id: str,
    finetuned_lora_path: str,
    base_local_dir: str = None,
    use_bfloat16: bool = False,
    add_chat_template: bool = True
):
    """
    Merges LoRA fine-tuned weights into a base model and saves the resulting model locally.

    This function supports:
    - Automatic memory cleanup before loading models.
    - Resizing token embeddings if vocabulary sizes mismatch.
    - Adding a chat template format (if missing) for chat-style models.

    Args:
        base_model_id (str): Hugging Face model ID or local path to the base model.
        finetuned_lora_path (str): Directory path containing LoRA adapter weights.
        base_local_dir (str, optional): Base directory where the merged model will be saved. 
            If None, a default path under `local/models_llora/` will be used.
        use_bfloat16 (bool, optional): If True, uses bfloat16 precision; otherwise uses float16. Defaults to False.
        add_chat_template (bool, optional): Whether to apply a chat template if the tokenizer lacks one. Defaults to True.

    Raises:
        RuntimeError: If model or tokenizer loading fails.
    """
    print("🧹 Cleaning up memory...")
    gc.collect()
    torch.cuda.empty_cache()

    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    print("🔄 Loading base tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load base model or tokenizer: {e}")

    # Ensure the model and tokenizer vocabularies are aligned
    vocab_size_tokenizer = len(tokenizer)
    vocab_size_model = model.get_input_embeddings().num_embeddings
    if vocab_size_tokenizer != vocab_size_model:
        print(f"⚠️ Resizing token embeddings: model ({vocab_size_model}) → tokenizer ({vocab_size_tokenizer})")
        model.resize_token_embeddings(vocab_size_tokenizer)

    # Optionally apply chat template formatting
    if add_chat_template:
        if getattr(tokenizer, "chat_template", None) is None:
            print("💬 Applying chat template format...")
            model, tokenizer = setup_chat_format(model, tokenizer)
        else:
            print("⚠️ Tokenizer already contains a chat_template. Skipping setup.")

    # Load and merge LoRA weights
    print(f"🔗 Loading LoRA fine-tuned weights from: {finetuned_lora_path}")
    try:
        model = PeftModel.from_pretrained(model, finetuned_lora_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load LoRA weights: {e}")

    print("🧠 Merging LoRA weights into the base model...")
    model = model.merge_and_unload()

    # Define save path
    base_model_name = base_model_id.split("/")[-1]
    merged_model_name = f"Orpo-{base_model_name}-FT"
    save_path = os.path.join(
        base_local_dir or os.path.join("..", "..", "..", "local", "models_llora"),
        merged_model_name
    )
    os.makedirs(save_path, exist_ok=True)

    # Save merged model and tokenizer
    print(f"💾 Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("✅ Merge complete! Model successfully saved locally.")
