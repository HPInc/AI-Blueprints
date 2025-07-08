import os
import sys
import torch
import logging
import pandas as pd
from tabulate import tabulate
from typing import List, Dict, Any, Optional, Tuple
from core.finetuning_inference.inference_runner import AcceleratedInferenceRunner
from core.selection.model_selection import ModelSelector
from datetime import datetime
from pathlib import Path

# Import path utilities from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class ModelComparer:
    """
    A utility for comparing outputs between a base model and a fine-tuned model locally.

    Attributes:
        prompts (list[str]): List of input prompts for inference.
        device (str): Device used for inference ("cuda" or "cpu").
        dtype (torch.dtype): Data type used during model inference.
        runner_base (AcceleratedInferenceRunner): Inference runner for the base model.
        runner_ft (AcceleratedInferenceRunner): Inference runner for the fine-tuned model.
        results (list): Collection of comparison results.
    """

    def __init__(
        self,
        base_selector: ModelSelector,
        finetuned_path: str,
        prompts: list[str],
        project_name: str = "model-comparison",  # Kept for API compatibility
        dtype=torch.float16
    ):
        """
        Initializes the ModelComparer.

        Args:
            base_selector (ModelSelector): Model selector for loading the base model.
            finetuned_path (str): Path to the fine-tuned model directory.
            prompts (list[str]): List of prompts to evaluate.
            project_name (str): Name for the comparison project (kept for compatibility).
            dtype (torch.dtype, optional): Data type for inference. Defaults to torch.float16.
        """
        self.prompts = prompts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.results = []

        # Initialize runners for base and fine-tuned models
        self.runner_base = AcceleratedInferenceRunner(
            model_selector=base_selector,
            dtype=dtype
        )

        self.runner_ft = AcceleratedInferenceRunner(
            model_selector=base_selector,
            finetuned_path=finetuned_path,
            dtype=dtype
        )

        self.runner_base.load_model()
        self.runner_ft.load_model()

    def compare(self):
        """
        Executes inference for each prompt with both the base and fine-tuned models,
        and displays a comparison of the results.

        Steps:
            - Run inference with the base model.
            - Run inference with the fine-tuned model.
            - Store both outputs for display.
            - Print a comparison table.

        Returns:
            List[Dict]: The comparison results for each prompt.
        """
        for idx, prompt in enumerate(self.prompts):
            print(f"⚙️ Running prompt {idx + 1}/{len(self.prompts)}")

            response_base = self.runner_base.infer(prompt)
            response_ft = self.runner_ft.infer(prompt)

            # Store results
            self.results.append({
                "prompt_id": idx,
                "prompt": prompt,
                "base_model_response": response_base,
                "finetuned_model_response": response_ft,
                "base_model_length": len(response_base),
                "finetuned_model_length": len(response_ft),
            })

        # Display comparison table
        self._display_comparison_table()
        
        print("✅ Finished comparing base and fine-tuned models.")
        return self.results

    def _display_comparison_table(self):
        """
        Displays a formatted table comparing the base and fine-tuned model responses.
        """
        table_data = []
        for idx, result in enumerate(self.results):
            # Format prompt and responses for display (truncate if too long)
            prompt_display = (result["prompt"][:50] + '...') if len(result["prompt"]) > 50 else result["prompt"]
            base_response = (result["base_model_response"][:100] + '...') if len(result["base_model_response"]) > 100 else result["base_model_response"]
            ft_response = (result["finetuned_model_response"][:100] + '...') if len(result["finetuned_model_response"]) > 100 else result["finetuned_model_response"]
            
            table_data.append([
                idx + 1,
                prompt_display,
                base_response,
                ft_response,
                result["base_model_length"],
                result["finetuned_model_length"]
            ])
        
        # Print the table
        headers = ["ID", "Prompt", "Base Model Response", "Fine-tuned Model Response", "Base Length", "FT Length"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        
    def save_results(self, output_path: Optional[str] = None):
        """
        Saves the comparison results to a CSV file.
        
        Args:
            output_path (str, optional): Path to save the results. 
                If None, saves to 'model_comparison_results.csv' in the current directory.
                
        Returns:
            str: Path to the saved file.
        """
        if not output_path:
            output_path = "model_comparison_results.csv"
            
        # Create a DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return output_path
