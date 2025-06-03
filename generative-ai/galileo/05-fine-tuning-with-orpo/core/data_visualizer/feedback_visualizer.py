import os
import logging
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, util

class UltraFeedbackVisualizer:
    """
    A class to log human feedback data from the UltraFeedback dataset into TensorBoard,
    enabling visual comparison between preferred (chosen) and rejected model responses.

    This tool highlights `score_chosen` vs. `score_rejected` differences through charts 
    and markdown summaries, making the results accessible even for non-technical audiences.

    Attributes:
        train_dataset (Dataset): Subset of UltraFeedback training samples.
        test_dataset (Dataset): Subset of UltraFeedback testing samples.
        log_dir (str): Directory path for saving TensorBoard logs.
        max_samples (int): Maximum number of samples to visualize from each dataset.
    """

    def __init__(self, train_dataset, test_dataset, log_dir="/phoenix/tensorboard/tensorlogs", max_samples=20):
        """
        Initializes the UltraFeedbackVisualizer.

        Args:
            train_dataset (Dataset): The training split of UltraFeedback.
            test_dataset (Dataset): The testing split of UltraFeedback.
            log_dir (str, optional): Directory to save TensorBoard logs. Defaults to "/phoenix/tensorboard/tensorlogs".
            max_samples (int, optional): Maximum number of examples to log per split. Defaults to 20.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.log_dir = log_dir
        self.max_samples = max_samples

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UltraFeedbackVisualizer")

    def _extract_text(self, field, field_name="unknown", idx=0):
        """
        Safely extracts and normalizes text from various formats (string, dictionary, list).

        Args:
            field (Union[str, dict, list]): The field containing the text.
            field_name (str, optional): Name of the field for logging purposes. Defaults to "unknown".
            idx (int, optional): Index of the example for debug logging. Defaults to 0.

        Returns:
            str: The extracted and normalized text string.
        """
        try:
            if isinstance(field, dict) and "text" in field:
                return field["text"]
            elif isinstance(field, list):
                return " ".join(
                    [f["text"] if isinstance(f, dict) and "text" in f else str(f) for f in field]
                )
            else:
                return str(field)
        except Exception as e:
            self.logger.error(f"[Example {idx}] ❌ Error parsing field '{field_name}': {e}")
            return "[ERROR]"

    def log_dataset(self, dataset, tag_prefix="train"):
        """
        Logs a subset of dataset examples into TensorBoard, including:
        - Scalar scores for chosen and rejected responses.
        - Delta score between chosen and rejected.
        - Markdown visual summaries for each example.

        Args:
            dataset (Dataset): The dataset split to log (e.g., train or test).
            tag_prefix (str, optional): Namespace prefix for TensorBoard tags. Defaults to "train".
        """
        score_chosen_list = []
        score_rejected_list = []
        score_delta_list = []

        for idx, example in enumerate(dataset.select(range(min(len(dataset), self.max_samples)))):
            try:
                prompt = self._extract_text(example["prompt"], "prompt", idx)
                chosen = self._extract_text(example["chosen"], "chosen", idx)
                rejected = self._extract_text(example["rejected"], "rejected", idx)
                score_chosen = example.get("score_chosen", 0.0)
                score_rejected = example.get("score_rejected", 0.0)
                delta = score_chosen - score_rejected

                score_chosen_list.append(score_chosen)
                score_rejected_list.append(score_rejected)
                score_delta_list.append(delta)

                # Log delta score
                self.writer.add_scalar(f"{tag_prefix}/ScoreDelta/Example_{idx}", delta, global_step=idx)

                # Log raw scores
                self.writer.add_scalar(f"{tag_prefix}/Score/Chosen", score_chosen, global_step=idx)
                self.writer.add_scalar(f"{tag_prefix}/Score/Rejected", score_rejected, global_step=idx)

                # Prepare markdown summary
                preferred_tag = "🟢 Chosen" if delta >= 0 else "🔴 Rejected"
                markdown_text = f"""
### ❓ Prompt:
{prompt}

---

### 🟢 Chosen (Score: {score_chosen:.1f}):
{chosen}

---

### 🔴 Rejected (Score: {score_rejected:.1f}):
{rejected}

---

🏁 **Human preference:** {preferred_tag}  
📉 **Score difference:** {delta:.2f} points
"""

                self.writer.add_text(f"{tag_prefix}/Example_{idx}/Visual", markdown_text, global_step=idx)
                self.logger.info(f"[Example {idx}] ✅ Logged successfully.")

            except Exception as e:
                self.logger.error(f"[Example {idx}] ❌ Failed to process example: {e}")

        # Log average scores
        if score_chosen_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_chosen", sum(score_chosen_list) / len(score_chosen_list))
        if score_rejected_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_rejected", sum(score_rejected_list) / len(score_rejected_list))
        if score_delta_list:
            self.writer.add_scalar(f"summary/{tag_prefix}_mean_delta", sum(score_delta_list) / len(score_delta_list))

    def run(self):
        """
        Executes the full logging workflow:
        - Logs training samples.
        - Logs testing samples.
        - Finalizes the TensorBoard writer.

        Launch TensorBoard to visualize the logs using:
        tensorboard --logdir=<log_dir> --port 6006
        """
        self.logger.info("📊 Logging training samples (human feedback)...")
        self.log_dataset(self.train_dataset, tag_prefix="train")

        self.logger.info("📊 Logging test samples (human feedback)...")
        self.log_dataset(self.test_dataset, tag_prefix="test")

        self.writer.close()
        self.logger.info(f"✅ Human feedback logging complete!\n"
                         f"To visualize, run:\n"
                         f"tensorboard --logdir={self.log_dir} --port 6006")
