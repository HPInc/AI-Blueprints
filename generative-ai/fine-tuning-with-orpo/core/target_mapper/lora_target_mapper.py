import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRATargetMapper")

class LoRATargetMapper:
    """
    Utility class to map model types to their corresponding target modules
    for LoRA (Low-Rank Adaptation) fine-tuning.

    This allows dynamic selection of modules based on the base model architecture 
    (e.g., Mistral, LLaMA, Gemma) for training scripts.

    It is particularly useful when applying QLoRA or PEFT fine-tuning 
    across different transformer-based architectures with shared components.

    Attributes:
        TARGET_MODULES_MAP (dict): Mapping of model name keywords to their respective target modules.
    """

    TARGET_MODULES_MAP = {
        "mistral": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "llama": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "gemma": ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    }

    @classmethod
    def get_target_modules(cls, model_id: str) -> list[str]:
        """
        Retrieves the target modules for LoRA fine-tuning based on the provided model ID.

        Args:
            model_id (str): Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf').

        Returns:
            list[str]: List of module names to target for LoRA adaptation.

        Raises:
            ValueError: If no matching target modules are defined for the given model_id.
        """
        model_id_lower = model_id.lower()
        for key in cls.TARGET_MODULES_MAP:
            if key in model_id_lower:
                logger.info(f"✅ Matched model '{model_id}' to LoRA target modules: {key}")
                return cls.TARGET_MODULES_MAP[key]
        logger.error(f"❌ No LoRA target modules defined for model: {model_id}")
        raise ValueError(f"No LoRA target_modules defined for model: {model_id}")