# Standard Library Imports
import logging

# Third-Party Libraries
import shutil

# MLflow for Experiment Tracking and Model Management
import mlflow
from mlflow import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature

# Transformers
from transformers import pipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configurations
MODEL_PERSONAL_NAME = "morgana-rodrigues/bert_qa"
EXPERIMENT_NAME = "BERT model for Q&A"
MODEL_NAME = "BERT_QA"

# Set up the chunk separator for text processing
CHUNK_SEPARATOR = "\n\n"

# Create a logger for this notebook
logger = logging.getLogger('deployment-notebook')
logger.info("Logging configured successfully")

class DistilBERTModel(mlflow.pyfunc.PythonModel):
    def _preprocess(self, inputs):
        """
        Preprocesses the input data.

        Args:
            inputs: A dictionary containing two keys:
                - 'context': A list with the context text.
                - 'question': A list with the question to be answered.

        Returns:
            tuple: A tuple containing the context (str) and the question (str).
        """
        try:
            context = inputs['context'][0]
            question = inputs['question'][0]
            logger.info("pre processing", context,question)
            return context, question
        except Exception as e:
            logger.error(f"Error preprocessing the input data: {str(e)}")  
        
    def load_context(self, context):
        """
        Loads the question-answering pipeline using the saved model artifact.

        Args:
            context: The MLflow context object 
                containing the loaded artifacts.
        """
        try:
            self.model = pipeline(
                'question-answering',
                model=context.artifacts["model"],
                device=0
            )
        except Exception as e:
            logger.error(f"Error loading the question-answering pipeline: {str(e)}")     
        
    def predict(self, context, model_input, params):
        """
        Runs inference using the loaded model and input data.

        Args:
            context: The MLflow context object 
                with access to artifacts.
            model_input: A dictionary containing 'context' and 'question' keys.

        Returns:
            The output from the model containing the predicted answer and optionally the score.
        """
        try:
            in_ctx, question = self._preprocess(model_input)
            output = self.model(context=in_ctx, question=question)
            return output
        except Exception as e:
            logger.error(f"Error running inference: {str(e)}")

    @classmethod
    def log_model(cls, model_name, source_trainer = None, source_pipeline = None, demo_folder="../demo"): 
        """
        Logs the model to MLflow, including artifacts, dependencies, and input/output signatures.

        Args:
            model_name: Path where the model will be temporarily saved before logging.
            source_trainer: A trainer object with a `.save_model()` method. Defaults to None.
            source_pipeline: A pipeline object with a `.save_pretrained()` method. Defaults to None.
            demo_folder: Path to the folder containing the compiled demo UI. Defaults to "demo".
        """
        try:
            input_schema = Schema(
                [
                    ColSpec("string", "context"),
                    ColSpec("string", "question"),
                ]
            )
            output_schema = Schema(
                [
                    ColSpec("string", "answer")
                ]
            )
            
            params_schema = ParamSchema(
                [
                    ParamSpec("show_score", "boolean", False)
                ]
            )
        
            signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)
            if source_trainer is not None:
                source_trainer.save_model(model_name)
            elif source_pipeline is not None:
                source_pipeline.save_pretrained(model_name)
                
            requirements = [
                "transformers==4.48.0",
                "tf_keras"
            ]
            mlflow.pyfunc.log_model(
                model_name,
                python_model=cls(),
                artifacts={"model": model_name, "demo": demo_folder},
                signature=signature,
                pip_requirements=requirements
            )
            shutil.rmtree(model_name)
            logger.info("Logging model to MLflow done successfully")

        except Exception as e:
            logger.error(f"Error logging model to MLflow: {str(e)}")   
       
if __name__ == "__main__":
    model_name = MODEL_PERSONAL_NAME

    qa_pipeline = pipeline(
      'question-answering',
      model=model_name,
      device=0 # -1 means running on CPU
    )
    
    mlflow.set_experiment(experiment_name = EXPERIMENT_NAME)
    with mlflow.start_run(run_name = MODEL_NAME) as run:
        logger.info(f"Run's Artifact URI: {run.info.artifact_uri}")
        DistilBERTModel.log_model(model_name = MODEL_NAME, source_pipeline=qa_pipeline)
        mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}", name = MODEL_NAME)
        logger.info("Model registred!")

