import torch
from transformers import BitsAndBytesConfig

class QuantizationSelector:
    """
    Automatically selects the optimal quantization strategy (4-bit or 8-bit)
    based on available GPU memory and number of GPUs.

    This class helps optimize memory usage and loading efficiency for large models
    by choosing either 8-bit quantization for high-resource setups or 4-bit quantization
    for more constrained environments.

    Attributes:
        vram_threshold_8bit (int): Minimum VRAM (in GB) required per GPU to enable 8-bit loading.
        min_gpus_for_8bit (int): Minimum number of GPUs required for 8-bit quantization eligibility.
        num_gpus (int): Number of detected GPUs.
        vram_list (list[float]): List of available VRAM per GPU (in GB).
    """

    def __init__(self, vram_threshold_8bit: int = 30, min_gpus_for_8bit: int = 2):
        """
        Initializes the QuantizationSelector.

        Args:
            vram_threshold_8bit (int, optional): VRAM (GB) threshold per GPU to allow 8-bit quantization. Defaults to 30.
            min_gpus_for_8bit (int, optional): Minimum number of GPUs needed for 8-bit quantization. Defaults to 2.
        """
        self.vram_threshold_8bit = vram_threshold_8bit
        self.min_gpus_for_8bit = min_gpus_for_8bit
        self.num_gpus = torch.cuda.device_count()
        self.vram_list = self._get_gpu_memory_list()

    def _get_gpu_memory_list(self) -> list:
        """
        Retrieves the total memory (in GB) for each detected GPU.

        Returns:
            list[float]: List of available VRAM per GPU.
        """
        return [
            torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(self.num_gpus)
        ]

    def _is_eligible_for_8bit(self) -> bool:
        """
        Determines if the current hardware setup qualifies for 8-bit quantization.

        Returns:
            bool: True if eligible for 8-bit quantization, False otherwise.
        """
        return (
            self.num_gpus >= self.min_gpus_for_8bit and
            min(self.vram_list) >= self.vram_threshold_8bit
        )

    def get_config(self) -> BitsAndBytesConfig:
        """
        Returns the appropriate BitsAndBytesConfig based on available resources.

        Returns:
            BitsAndBytesConfig: Configuration object for model quantization.

        Notes:
            - If the setup meets the requirements, 8-bit quantization is selected.
            - Otherwise, 4-bit quantization (nf4) is used as fallback.
        """
        if self._is_eligible_for_8bit():
            print("✅ Using 8-bit quantization (sufficient GPUs and VRAM available).")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            print("⚠️ Using 4-bit quantization (fallback due to lower resources).")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
