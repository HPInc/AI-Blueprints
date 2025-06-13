import os
import sys
import torch
import logging
import promptquality as pq
from core.finetuning_inference.inference_runner import AcceleratedInferenceRunner
from core.selection.model_selection import ModelSelector
from datetime import datetime

# Ensure src directory is in the system path
src_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.utils import initialize_galileo_evaluator

class GalileoLocalComparer:
    """
    A utility for comparing outputs between a base model and a fine-tuned model locally,
    and logging the results into Galileo's EvaluateRun.

    Attributes:
        prompts (list[str]): List of input prompts for inference.
        project_name (str): Name of the Galileo project.
        device (str): Device used for inference ("cuda" or "cpu").
        dtype (torch.dtype): Data type used during model inference.
        runner_base (AcceleratedInferenceRunner): Inference runner for the base model.
        runner_ft (AcceleratedInferenceRunner): Inference runner for the fine-tuned model.
        evaluate_run (EvaluateRun): Galileo EvaluateRun instance for logging results.
    """

    def __init__(
        self,
        base_selector: ModelSelector,
        finetuned_path: str,
        prompts: list[str],
        galileo_project_name: str,
        dtype=torch.float16
    ):
        """
        Initializes the GalileoLocalComparer.

        Args:
            base_selector (ModelSelector): Model selector for loading the base model.
            finetuned_path (str): Path to the fine-tuned model directory.
            prompts (list[str]): List of prompts to evaluate.
            galileo_project_name (str): Name of the Galileo project to log results into.
            dtype (torch.dtype, optional): Data type for inference. Defaults to torch.float16.
        """
        self.prompts = prompts
        self.project_name = galileo_project_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

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

        # Initialize EvaluateRun (for logging comparisons)
        self.evaluate_run = pq.EvaluateRun(
            run_name=f"compare-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
            project_name=self.project_name,
            scorers=[
                pq.Scorers.correctness,
                pq.Scorers.context_adherence_luna,
                pq.Scorers.instruction_adherence_plus,
                pq.Scorers.chunk_attribution_utilization_luna,
            ]
        )

    def compare(self):
        """
        Executes inference for each prompt with both the base and fine-tuned models,
        and logs the results into Galileo EvaluateRun.

        Steps:
            - Run inference with the base model.
            - Run inference with the fine-tuned model.
            - Log both outputs, along with metadata.
            - Finalize the EvaluateRun.

        Returns:
            None
        """
        for idx, prompt in enumerate(self.prompts):
            print(f"⚙️ Running prompt {idx + 1}/{len(self.prompts)}")

            response_base = self.runner_base.infer(prompt)
            response_ft = self.runner_ft.infer(prompt)

            # Log base model output
            self.evaluate_run.add_single_step_workflow(
                input=prompt,
                output=response_base,
                model="BASE_MODEL_LOCAL",
                metadata={"example_id": str(idx), "type": "base"},
                duration_ns=2000,
            )

            # Log fine-tuned model output
            self.evaluate_run.add_single_step_workflow(
                input=prompt,
                output=response_ft,
                model="FINETUNED_MODEL_LOCAL",
                metadata={"example_id": str(idx), "type": "fine-tuned"},
                duration_ns=2000,
            )

        # Finalize the EvaluateRun session
        self.evaluate_run.finish()

        print("✅ Finished logging outputs for both models to Galileo.")
