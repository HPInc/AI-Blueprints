import mlflow
from mlflow import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature
from transformers import pipeline
import torch
import json
import os

class DistilBERTModel(mlflow.pyfunc.PythonModel):
    def _preprocess(self, inputs):
        context = inputs['context'][0]
        question = inputs['question'][0]
        print("pre processing", context,question)
        return context, question
        
    def load_context(self, context):
        self.model = pipeline(
            'question-answering',
             model=context.artifacts["model"],
             device=0
        )
        
    def predict(self, context, model_input, params):
        in_ctx, question = self._preprocess(model_input)
        output = self.model(context=in_ctx, question=question)
        return output

    @classmethod
    def log_model(cls, model_name, source_trainer = None, source_pipeline = None, demo_folder="demo"): 
        import shutil
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
            "transformers==4.47.0",
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
       
if __name__ == "__main__":
    model_name = "morgana-rodrigues/bert_qa"

    qa_pipeline = pipeline(
      'question-answering',
      model=model_name,
      device=0 # -1 means running on CPU
    )
    
    mlflow.set_experiment(experiment_name='BERT Model for Q&A')
    with mlflow.start_run(run_name='BERT_QA') as run:
        print(f"Run's Artifact URI: {run.info.artifact_uri}")
        DistilBERTModel.log_model(model_name='BERT_QA', source_pipeline=qa_pipeline)
        mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/BERT_QA", name='BERT_QA')

